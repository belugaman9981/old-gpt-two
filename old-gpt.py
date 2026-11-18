"""old_gpt v2.0 - a tiny GPT that trains on your own text AND keeps learning.

What's new vs v1 (character-level, 3M params, CPU only):
  * Byte-level tokens (vocab 256): any text/language works, nothing is ever "unknown",
    so the model can keep learning new material later without a vocab change.
  * Modern architecture: RMSNorm, rotary position embeddings (RoPE), SwiGLU MLP,
    no biases, tied embeddings, dropout.
  * Better training: AdamW with weight-decay groups, warmup + cosine LR, grad clipping,
    bf16/fp16 on GPU, best-checkpoint saving, resume, optional torch.compile.
  * Continual learning: `learn` / `/teach` fine-tune on NEW text while replaying your
    old data, measure forgetting, and automatically revert if it forgets too much.
  * Better sampling: temperature + top-k + top-p, interactive chat REPL.
  * Presets: tiny (CPU) / small / base (GPU). `auto` picks one for your hardware.

Setup:    pip install "torch>=2.3"
Data:     put .txt / .md files in ./data  (aim for 1 MB+; Gutenberg headers are stripped)

Usage:
  python old_gpt_v2.py train                         # train from scratch (auto preset)
  python old_gpt_v2.py train --preset small --iters 8000
  python old_gpt_v2.py train --resume --iters 3000   # keep training the saved model
  python old_gpt_v2.py gen "Whan that Aprille"       # sample
  python old_gpt_v2.py learn notes.txt more.md       # learn new files without forgetting
  python old_gpt_v2.py learn --text "Matt's dog is named Max."
  python old_gpt_v2.py chat                          # interactive; /teach to learn live
  python old_gpt_v2.py rollback                      # undo the last save

Note: v1 checkpoints (old_gpt.pt, char-level) are not compatible - retrain with `train`.
"""
import argparse, contextlib, glob, math, os, random, shutil, sys, time
import re
import torch, torch.nn as nn, torch.nn.functional as F

CKPT = "old_gpt_v2.pt"

PRESETS = {  # model shape            (params: ~2.7M / ~14M / ~38M)
    "tiny":  dict(block=128, n_embd=192, n_head=6, n_layer=6,  dropout=0.10),
    "small": dict(block=256, n_embd=384, n_head=6, n_layer=8,  dropout=0.10),
    "base":  dict(block=512, n_embd=512, n_head=8, n_layer=12, dropout=0.10),
}
TRAIN_DEFAULTS = {
    "tiny":  dict(batch=32, lr=1e-3, iters=5000),
    "small": dict(batch=32, lr=6e-4, iters=8000),
    "base":  dict(batch=16, lr=3e-4, iters=15000),
}

# ----------------------------------------------------------------------------- data


def clean(t):
    """Normalise newlines and strip Project Gutenberg header/footer if present."""
    t = t.replace("\r\n", "\n")
    m = re.search(r"\*\*\* ?START OF[^\n]*\*\*\*", t)
    if m:
        t = t[m.end():]
    e = re.search(r"\*\*\* ?END OF", t)
    if e:
        t = t[:e.start()]
    return t.strip()


def read_files(paths):
    parts = []
    for f in paths:
        t = clean(open(f, encoding="utf-8", errors="ignore").read())
        print(f"loaded {f}: {len(t):,} chars")
        parts.append(t)
    return "\n\n".join(parts)


def load_text(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.md")))
    return read_files(files)


def encode(text):
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def decode(ids):
    return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class Corpus:
    """Train/val split of a token stream. Tiny texts are tiled so they can still be batched."""

    def __init__(self, ids, block, val_frac=0.05):
        need = block * 8
        if len(ids) < need:
            ids = ids.repeat(math.ceil(need / len(ids)))
        cut = int(len(ids) * (1 - val_frac))
        self.tr, self.va, self.block = ids[:cut], ids[cut:], block
        if len(self.va) <= block + 1:
            self.va = self.tr

    def batch(self, split, bs, dev):
        d = self.tr if split == "train" else self.va
        ix = torch.randint(len(d) - self.block - 1, (bs,))
        x = torch.stack([d[i:i + self.block] for i in ix])
        y = torch.stack([d[i + 1:i + self.block + 1] for i in ix])
        return x.to(dev), y.to(dev)


# ----------------------------------------------------------------------------- model


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.w).type_as(x)


def rope_cache(T, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(T).float(), inv)  # (T, hd/2)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: (B, H, T, hd)
    T = x.shape[2]
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    c, s = cos[:T], sin[:T]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["n_embd"]
        self.h, self.p = cfg["n_head"], cfg["dropout"]
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [z.view(B, T, self.h, C // self.h).transpose(1, 2) for z in (q, k, v)]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        return self.o(a.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = ((int(8 * d / 3) + 63) // 64) * 64
        self.w1, self.w3, self.w2 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["n_embd"]
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn, self.mlp = Attention(cfg), SwiGLU(d)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.n1(x), cos, sin))
        return x + self.drop(self.mlp(self.n2(x)))


class GPT(nn.Module):
    VOCAB = 256  # raw bytes

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["n_embd"]
        self.tok = nn.Embedding(self.VOCAB, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, self.VOCAB, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        for n, p in self.named_parameters():  # scaled init on residual output projections
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg["n_layer"]))
        self.head.weight = self.tok.weight  # weight tying
        cos, sin = rope_cache(cfg["block"], d // cfg["n_head"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=50, top_p=0.95):
        was_training = self.training
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -self.cfg["block"]:])
            l = logits[:, -1].float() / max(temp, 1e-5)
            if top_k:
                v, _ = torch.topk(l, min(top_k, l.size(-1)))
                l = l.masked_fill(l < v[:, [-1]], float("-inf"))
            if top_p < 1.0:
                sl, si = torch.sort(l, descending=True)
                pr = F.softmax(sl, dim=-1)
                sl = sl.masked_fill(pr.cumsum(-1) - pr > top_p, float("-inf"))
                l = torch.full_like(l, float("-inf")).scatter(1, si, sl)
            idx = torch.cat([idx, torch.multinomial(F.softmax(l, dim=-1), 1)], dim=1)
        self.train(was_training)
        return idx


def sample(model, prompt, dev, n=600, temp=0.8, top_k=50, top_p=0.95):
    ids = list(prompt.encode("utf-8")) or [10]
    idx = torch.tensor([ids], device=dev)
    return decode(model.generate(idx, n, temp, top_k, top_p)[0].tolist())


# ----------------------------------------------------------------------------- training


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def amp_setup(dev):
    """Returns (autocast factory, dtype). bf16 on modern GPUs, fp16+scaler on old ones, fp32 elsewhere."""
    if dev == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return (lambda: torch.autocast("cuda", dtype=dt)), dt
    return contextlib.nullcontext, None


@torch.no_grad()
def evaluate(model, corpus, bs, dev, n=20):
    ctx, _ = amp_setup(dev)
    was_training = model.training
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y = corpus.batch("val", bs, dev)
        with ctx():
            tot += model(x, y)[1].item()
    model.train(was_training)
    return tot / n


def lr_at(it, iters, lr, warmup):
    if it < warmup:
        return lr * (it + 1) / warmup
    p = (it - warmup) / max(1, iters - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))  # cosine down to 10%


def fit(model, sources, probs, iters, lr, bs, dev, warmup=100, wd=0.1,
        eval_every=0, eval_corpus=None, on_best=None, log_every=0):
    ctx, dt = amp_setup(dev)
    scaler = torch.amp.GradScaler("cuda") if dt == torch.float16 else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": wd},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=(dev == "cuda"))
    model.train()
    t0, run, best = time.time(), None, float("inf")
    for it in range(iters + 1):
        if eval_every and (it % eval_every == 0 or it == iters):
            v = evaluate(model, eval_corpus, bs, dev)
            tr = f"{run:.3f}" if run is not None else "-"
            print(f"iter {it:5d} | train {tr} | val {v:.3f} ({v / math.log(2):.3f} bits/byte) | {time.time() - t0:.0f}s")
            if v < best:
                best = v
                if on_best:
                    on_best(v)
        elif log_every and it % log_every == 0 and run is not None:
            print(f"  step {it}/{iters} | loss {run:.3f}")
        if it == iters:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it, iters, lr, warmup)
        x, y = random.choices(sources, probs)[0].batch("train", bs, dev)
        with ctx():
            loss = model(x, y)[1]
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
    return best


# ----------------------------------------------------------------------------- continual learning


def learn(model, new_ids, old, cfg, dev, iters=300, lr=2e-4, bs=16, new_frac=0.5,
          max_forget=0.25, force=False):
    """Fine-tune on new text while replaying old data. Reverts if old-data loss rises too much."""
    new = Corpus(new_ids, cfg["block"])
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    b_new = evaluate(model, new, bs, dev, n=10)
    b_old = evaluate(model, old, bs, dev, n=10) if old else None
    sources, probs = ([new, old], [new_frac, 1 - new_frac]) if old else ([new], [1.0])
    fit(model, sources, probs, iters, lr, bs, dev, warmup=max(1, min(20, iters // 5)), log_every=100)
    a_new = evaluate(model, new, bs, dev, n=10)
    a_old = evaluate(model, old, bs, dev, n=10) if old else None
    reverted = bool(old) and (a_old - b_old) > max_forget and not force
    if reverted:
        model.load_state_dict(snap)
    return dict(b_new=b_new, a_new=a_new, b_old=b_old, a_old=a_old, reverted=reverted)


def report(r, max_forget):
    s = f"new text loss {r['b_new']:.3f} -> {r['a_new']:.3f}"
    if r["b_old"] is not None:
        s += f" | old data loss {r['b_old']:.3f} -> {r['a_old']:.3f}"
    else:
        s += " | (no ./data found, so forgetting can't be measured)"
    if r["reverted"]:
        s += (f"\n  !! old-data loss rose by more than {max_forget} - it was forgetting too much, so I REVERTED. "
              f"Try a lower --lr, fewer --iters, or a smaller --new-frac (or --force to keep it).")
    return s


# ----------------------------------------------------------------------------- checkpoints


def save_ckpt(path, model, cfg, meta, backup=True):
    if backup and os.path.exists(path):
        shutil.copyfile(path, path + ".prev")  # one-step undo
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "cfg": cfg, "meta": meta}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, dev):
    if not os.path.exists(path):
        sys.exit(f"No checkpoint at {path}. Run `python old_gpt_v2.py train` first.")
    ck = torch.load(path, map_location=dev, weights_only=True)
    model = GPT(ck["cfg"]).to(dev)
    model.load_state_dict(ck["model"])
    return model, ck["cfg"], ck["meta"]


def try_old_corpus(a, block):
    ids = encode(load_text(a.data)) if os.path.isdir(a.data) else None
    return Corpus(ids, block) if ids is not None and len(ids) > block * 8 else None


# ----------------------------------------------------------------------------- commands


def cmd_train(a):
    dev = pick_device(a.device)
    preset = a.preset if a.preset != "auto" else ("small" if dev == "cuda" else "tiny")
    td = TRAIN_DEFAULTS[preset]
    iters, bs, lr = a.iters or td["iters"], a.batch or td["batch"], a.lr or td["lr"]

    ids = encode(load_text(a.data))
    if len(ids) < 50_000:
        sys.exit(f"Only {len(ids):,} bytes found in {a.data}/. Put more .txt/.md files there (aim for 1 MB+).")

    if a.resume:
        model, cfg, meta = load_ckpt(a.ckpt, dev)
        print(f"resuming from {a.ckpt} (preset ignored, using saved shape)")
    else:
        cfg = dict(PRESETS[preset])
        if a.dropout is not None:
            cfg["dropout"] = a.dropout
        model, meta = GPT(cfg).to(dev), {"steps": 0, "learned": []}
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device {dev} | preset {preset} | {len(ids):,} bytes | {n_params:.1f}M parameters | {iters} iters")

    if os.path.exists(a.ckpt):
        shutil.copyfile(a.ckpt, a.ckpt + ".prev")  # back up once, before this run
    corpus = Corpus(ids, cfg["block"])
    fwd = model
    if a.compile:
        try:
            fwd = torch.compile(model)
        except Exception as e:
            print("torch.compile unavailable, continuing without:", e)

    def on_best(v):
        save_ckpt(a.ckpt, model, cfg, meta, backup=False)

    fit(fwd, [corpus], [1.0], iters, lr, bs, dev, warmup=min(100, iters // 10),
        eval_every=a.eval_every, eval_corpus=corpus, on_best=on_best)
    meta["steps"] = meta.get("steps", 0) + iters
    ck = torch.load(a.ckpt, map_location=dev, weights_only=True)  # re-save best weights with updated meta
    model.load_state_dict(ck["model"])
    save_ckpt(a.ckpt, model, cfg, meta, backup=False)
    print(f"\nbest checkpoint saved to {a.ckpt}\n--- sample ---\n{sample(model, chr(10), dev)}")


def cmd_gen(a):
    dev = pick_device(a.device)
    model, _, _ = load_ckpt(a.ckpt, dev)
    print(sample(model, a.prompt, dev, a.n, a.temp, a.top_k, a.top_p))


def cmd_learn(a):
    dev = pick_device(a.device)
    model, cfg, meta = load_ckpt(a.ckpt, dev)
    text = ""
    if a.files:
        text += read_files(a.files)
    if a.text:
        text += ("\n\n" if text else "") + a.text
    if not text.strip():
        sys.exit("Give me something to learn: files and/or --text \"...\"")
    old = try_old_corpus(a, cfg["block"])
    if not old:
        print("warning: no replay data in ./data - the model may forget what it knew.")
    r = learn(model, encode(text), old, cfg, dev, a.iters, a.lr, a.batch, a.new_frac, a.max_forget, a.force)
    print(report(r, a.max_forget))
    if r["reverted"]:
        return
    meta["learned"] = meta.get("learned", []) + [(a.files or ["--text"])[0]]
    meta["steps"] = meta.get("steps", 0) + a.iters
    save_ckpt(a.ckpt, model, cfg, meta)
    print(f"saved to {a.ckpt} (previous version kept as {a.ckpt}.prev)")


def cmd_chat(a):
    dev = pick_device(a.device)
    model, cfg, meta = load_ckpt(a.ckpt, dev)
    old = try_old_corpus(a, cfg["block"])
    temp, dirty = a.temp, False
    print("Type a prompt and I'll continue it.\n"
          "  /teach <text or @file>   learn something new right now (replays old data to avoid forgetting)\n"
          "  /temp 0.7                change sampling temperature\n"
          "  /save                    keep what I learned    /quit  exit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if cmd == "quit":
                break
            elif cmd == "temp":
                try:
                    temp = float(rest)
                except ValueError:
                    print("usage: /temp 0.8")
            elif cmd == "save":
                save_ckpt(a.ckpt, model, cfg, meta)
                dirty = False
                print("saved.")
            elif cmd == "teach" and rest:
                text = clean(open(rest[1:], encoding="utf-8", errors="ignore").read()) if rest.startswith("@") else rest
                r = learn(model, encode(text), old, cfg, dev, iters=a.teach_iters, lr=a.lr,
                          bs=16, new_frac=a.new_frac, max_forget=a.max_forget)
                print(report(r, a.max_forget))
                dirty = dirty or not r["reverted"]
            else:
                print("commands: /teach <text|@file>, /temp N, /save, /quit")
        else:
            print(sample(model, line, dev, a.n, temp, a.top_k, a.top_p))
    if dirty and input("Save what I learned this session? [y/N] ").strip().lower() == "y":
        save_ckpt(a.ckpt, model, cfg, meta)
        print("saved.")


def cmd_rollback(a):
    prev = a.ckpt + ".prev"
    if not os.path.exists(prev):
        sys.exit("No .prev backup to roll back to.")
    shutil.copyfile(prev, a.ckpt)
    print(f"restored {a.ckpt} from {prev}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", default=CKPT)
    common.add_argument("--data", default="data")
    common.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    common.add_argument("--seed", type=int, default=1)

    gen_args = argparse.ArgumentParser(add_help=False)
    gen_args.add_argument("-n", type=int, default=600, help="bytes to generate")
    gen_args.add_argument("--temp", type=float, default=0.8)
    gen_args.add_argument("--top-k", type=int, default=50)
    gen_args.add_argument("--top-p", type=float, default=0.95)

    learn_args = argparse.ArgumentParser(add_help=False)
    learn_args.add_argument("--lr", type=float, default=2e-4)
    learn_args.add_argument("--new-frac", type=float, default=0.5, help="share of batches from the new text (rest = replay)")
    learn_args.add_argument("--max-forget", type=float, default=0.25, help="revert if old-data loss rises more than this")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", parents=[common], help="train from scratch (or --resume)")
    t.add_argument("--preset", default="auto", choices=["auto", *PRESETS])
    t.add_argument("--iters", type=int)
    t.add_argument("--batch", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--dropout", type=float)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--compile", action="store_true", help="torch.compile (Linux/CUDA; skip on Windows)")
    t.set_defaults(fn=cmd_train)

    g = sub.add_parser("gen", parents=[common, gen_args], help="sample from the saved model")
    g.add_argument("prompt", nargs="?", default="\n")
    g.set_defaults(fn=cmd_gen)

    l = sub.add_parser("learn", parents=[common, learn_args], help="learn new files/text without forgetting")
    l.add_argument("files", nargs="*")
    l.add_argument("--text")
    l.add_argument("--iters", type=int, default=300)
    l.add_argument("--batch", type=int, default=16)
    l.add_argument("--force", action="store_true", help="keep the result even if it forgot too much")
    l.set_defaults(fn=cmd_learn)

    c = sub.add_parser("chat", parents=[common, gen_args, learn_args], help="interactive prompt + live /teach")
    c.add_argument("--teach-iters", type=int, default=60)
    c.set_defaults(fn=cmd_chat)

    r = sub.add_parser("rollback", parents=[common], help="restore the previous checkpoint")
    r.set_defaults(fn=cmd_rollback)

    a = p.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    a.fn(a)


if __name__ == "__main__":
    main() 


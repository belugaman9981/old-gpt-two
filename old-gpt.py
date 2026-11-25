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
  * Better sampling: temperature + top-k + top-p + repetition control, interactive chat REPL.
  * Presets: tiny (CPU) / small / base (GPU). `auto` picks one for your hardware.

Setup:    pip install "torch>=2.3"
Data:     put .txt / .md files in ./data  (aim for 1 MB+; Gutenberg headers are stripped)

Usage:
  python old_gpt_v2.py train                         # train from scratch (auto preset)
  python old_gpt_v2.py train --preset small --iters 8000
  python old_gpt_v2.py train --resume --iters 3000   # keep training the saved model
  python old_gpt_v2.py train --threads 4 --batch 128 --iters 2500 --low-priority   # gentle on CPU, bigger batches in RAM
  python old_gpt_v2.py gen "Whan that Aprille"       # sample
  python old_gpt_v2.py learn notes.txt more.md       # learn new files without forgetting
  python old_gpt_v2.py learn --text "Matt's dog is named Max."
  python old_gpt_v2.py chat                          # interactive; /teach to learn live
  python old_gpt_v2.py rollback                      # undo the last save

Note: v1 checkpoints (old_gpt.pt, char-level) are not compatible - retrain with `train`.
"""
import argparse, contextlib, glob, math, os, random, shutil, sys, time
import re
from collections import Counter
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
    "base":  dict(batch=24, lr=3e-4, iters=15000),
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


DATA_EXTS = (".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".java", ".c", ".cpp", ".h")


def load_text(data_dir):
    files = set()
    for ext in DATA_EXTS:
        files.update(glob.glob(os.path.join(data_dir, "**", "*" + ext), recursive=True))
    return read_files(sorted(files))


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
        pos = ix[:, None] + torch.arange(self.block)  # one gather instead of a Python loop
        return d[pos].to(dev), d[pos + 1].to(dev)


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
    def generate(self, idx, n, temp=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.15, frequency_penalty=0.0):
        was_training = self.training
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -self.cfg["block"]:])
            l = logits[:, -1].float() / max(temp, 1e-5)
            if repetition_penalty != 1.0 or frequency_penalty:
                l = penalize_repetition(
                    l, idx, repetition_penalty, frequency_penalty)
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


def sample(model, prompt, dev, n=600, temp=0.8, top_k=50, top_p=0.95,
           repetition_penalty=1.15, frequency_penalty=0.0):
    ids = list(prompt.encode("utf-8")) or [10]
    idx = torch.tensor([ids], device=dev)
    return decode(model.generate(
        idx, n, temp, top_k, top_p, repetition_penalty, frequency_penalty
    )[0].tolist())


def penalize_repetition(logits, ids, repetition_penalty=1.15,
                        frequency_penalty=0.0):
    """Reduce the chance of repeating tokens already present in the context."""
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than 0.")
    if frequency_penalty < 0:
        raise ValueError("frequency_penalty must not be negative.")
    if repetition_penalty != 1.0:
        seen = torch.zeros_like(logits, dtype=torch.bool)
        seen.scatter_(1, ids, True)
        positive = logits >= 0
        logits = torch.where(
            seen & positive, logits / repetition_penalty,
            torch.where(seen, logits * repetition_penalty, logits))
    if frequency_penalty:
        counts = torch.zeros_like(logits)
        counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=logits.dtype))
        logits = logits - counts * frequency_penalty
    return logits


# ----------------------------------------------------------------------------- training


def lower_priority():
    """Run politely in the background so the PC stays responsive."""
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:
        pass


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_ram_gb():
    if os.name == "nt":
        try:
            import ctypes
            kb = ctypes.c_ulonglong()
            if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kb)):
                return kb.value / (1024 * 1024)
        except Exception:
            pass
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 0.0


def pick_auto_preset(dev, preset):
    if preset != "auto":
        return preset
    if dev == "cuda":
        try:
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram >= 24:
                return "base"
        except Exception:
            pass
        return "small"
    if dev == "mps":
        return "small"
    cores = os.cpu_count() or 1
    ram = available_ram_gb()
    if ram >= 24 and cores >= 12:
        return "base"
    if ram >= 8 and cores >= 8:
        return "small"
    return "tiny"


def safe_cpu_threads():
    cores = os.cpu_count() or 1
    if cores <= 4:
        return cores
    return min(8, max(1, cores - 4))


def amp_setup(dev):
    """Returns (autocast factory, dtype). bf16 on modern GPUs, fp16+scaler on old ones, fp32 elsewhere."""
    if dev == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return (lambda: torch.autocast("cuda", dtype=dt)), dt
    return contextlib.nullcontext, None


@torch.no_grad()
def evaluate(model, corpus, bs, dev, n=20, batches=None):
    ctx, _ = amp_setup(dev)
    was_training = model.training
    model.eval()
    tot = 0.0
    if batches is None:
        batches = [corpus.batch("val", bs, "cpu") for _ in range(n)]
    for x, y in batches:
        x, y = x.to(dev), y.to(dev)
        with ctx():
            tot += model(x, y)[1].item()
    model.train(was_training)
    return tot / len(batches)


def lr_at(it, iters, lr, warmup):
    if it < warmup:
        return lr * (it + 1) / warmup
    p = (it - warmup) / max(1, iters - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))  # cosine down to 10%


def training_batch(sources, probs, bs, dev, split="train"):
    if len(sources) == 1:
        return sources[0].batch(split, bs, dev)
    expected = bs * probs[0]
    new_count = int(expected)
    if random.random() < expected - new_count:
        new_count += 1
    batches = [source.batch(split, count, dev)
               for source, count in zip(sources, (new_count, bs - new_count)) if count]
    return tuple(torch.cat(parts, dim=0) for parts in zip(*batches))


class LessonCorpus:
    """Sample lessons equally, rather than letting long lessons dominate replay."""

    def __init__(self, texts, block):
        self.block = block
        self.lessons = [Corpus(encode(text), block) for text in texts]

    def add(self, text):
        self.lessons.append(Corpus(encode(text), self.block))

    def batch(self, split, bs, dev):
        counts = Counter(random.choices(range(len(self.lessons)), k=bs))
        batches = [self.lessons[i].batch(split, count, dev) for i, count in counts.items()]
        return tuple(torch.cat(parts, dim=0) for parts in zip(*batches))


class ReplayCorpus:
    def __init__(self, data, texts, block):
        self.data = data
        self.lessons = LessonCorpus(texts, block)

    def add(self, text):
        self.lessons.add(text)

    def batch(self, split, bs, dev):
        sources = ([self.data] if self.data is not None else [])
        if self.lessons.lessons:
            sources.append(self.lessons)
        return training_batch(sources, [0.5, 0.5], bs, dev, split)


def fit(model, sources, probs, iters, lr, bs, dev, warmup=100, wd=0.1,
        eval_every=0, eval_corpus=None, on_best=None, log_every=0, patience=0, state=None):
    ctx, dt = amp_setup(dev)
    scaler = torch.amp.GradScaler("cuda") if dt == torch.float16 else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": wd},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=(dev == "cuda"))
    model.train()
    t0, run, best, bad = time.time(), None, float("inf"), 0
    for it in range(iters + 1):
        if state is not None:
            state["it"] = it
        if eval_every and (it % eval_every == 0 or it == iters):
            v = evaluate(model, eval_corpus, bs, dev)
            tr = f"{run:.3f}" if run is not None else "-"
            print(f"iter {it:5d} | train {tr} | val {v:.3f} ({v / math.log(2):.3f} bits/byte) | {time.time() - t0:.0f}s")
            if v < best:
                best, bad = v, 0
                if on_best:
                    on_best(v)
            else:
                bad += 1
                if patience and bad >= patience:
                    print(f"validation loss hasn't improved for {bad} checks - stopping early "
                          f"(more training would just memorise the data; add more text instead).")
                    break
        elif log_every and it % log_every == 0 and run is not None:
            elapsed = time.time() - t0
            remaining = elapsed * (iters - it) / it
            print(f"  step {it}/{iters} | smoothed loss {run:.3f} | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining", flush=True)
        if it == iters:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it, iters, lr, warmup)
        x, y = training_batch(sources, probs, bs, dev)
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
    if not len(new_ids):
        raise ValueError("Cannot learn from empty text.")
    if not 0 <= new_frac <= 1:
        raise ValueError("new_frac must be between 0 and 1.")
    new = Corpus(new_ids, cfg["block"])
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    new_batches = [new.batch("val", bs, "cpu") for _ in range(10)]
    old_batches = [old.batch("val", bs, "cpu") for _ in range(10)] if old else None
    b_new = evaluate(model, new, bs, dev, batches=new_batches)
    b_old = evaluate(model, old, bs, dev, batches=old_batches) if old else None
    sources, probs = ([new, old], [new_frac, 1 - new_frac]) if old else ([new], [1.0])
    fit(model, sources, probs, iters, lr, bs, dev, warmup=max(1, min(20, iters // 5)),
        log_every=max(1, min(100, iters // 10)))
    a_new = evaluate(model, new, bs, dev, batches=new_batches)
    a_old = evaluate(model, old, bs, dev, batches=old_batches) if old else None
    reverted = bool(old) and (a_old - b_old) > max_forget and not force
    if reverted:
        model.load_state_dict(snap)
    return dict(b_new=b_new, a_new=a_new, b_old=b_old, a_old=a_old, reverted=reverted)


def report(r, max_forget):
    s = f"new text loss {r['b_new']:.3f} -> {r['a_new']:.3f}"
    if r["b_old"] is not None:
        s += f" | old data loss {r['b_old']:.3f} -> {r['a_old']:.3f}"
    else:
        s += " | (no replay data available, so forgetting can't be measured)"
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


def replay_corpus(data, meta, block):
    texts = meta.get("lesson_texts", [])
    return ReplayCorpus(data, texts, block) if data is not None or texts else None


def lesson_summary(text, limit=70):
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[:limit - 3] + "..."


def print_lessons(meta):
    lessons = meta.get("lesson_texts", [])
    if not lessons:
        print("No saved lessons.")
        return
    print(f"{len(lessons)} saved lesson(s):")
    for i, text in enumerate(lessons, 1):
        print(f"  {i}. {lesson_summary(text)}")


def print_history(meta):
    history = meta.get("experiments", [])
    if not history:
        print("No learning history.")
        return
    print(f"{len(history)} learning run(s):")
    for i, entry in enumerate(history, 1):
        status = "reverted" if entry["reverted"] else "accepted"
        print(
            f"  {i}. {entry['timestamp']} | {status} | {entry['iters']} iters "
            f"| lr {entry['lr']:.1e} | new {entry['new_frac']:.2f} "
            f"| new loss {entry['b_new']:.3f}->{entry['a_new']:.3f}"
            f"| old loss "
            f"{'-' if entry['b_old'] is None else f'{entry['b_old']:.3f}->{entry['a_old']:.3f}'}"
        )


def checkpoint_status(path, model, cfg, meta):
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    lessons = meta.get("lesson_texts", [])
    experiments = meta.get("experiments", [])
    accepted = sum(not entry.get("reverted", False) for entry in experiments)
    reverted = sum(entry.get("reverted", False) for entry in experiments)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"checkpoint: {path}")
    print(f"size: {size / (1024 * 1024):.2f} MB")
    ram = available_ram_gb()
    print(f"hardware: {(ram / 1.0):.1f} GB RAM" if ram else "hardware: RAM unknown",
          f"| {os.cpu_count() or 1} logical CPUs")
    print(f"model: {n_params:.1f}M parameters | {cfg['n_layer']} layers "
          f"| {cfg['n_embd']} embedding width | context {cfg['block']} bytes")
    print(f"training steps: {meta.get('steps', 0):,}")
    print(f"saved lessons: {len(lessons)}")
    print(f"learning history: {len(experiments)} runs "
          f"| {accepted} accepted | {reverted} reverted")
    if experiments:
        last = experiments[-1]
        print(f"last run: {last['timestamp']} | "
              f"{'reverted' if last['reverted'] else 'accepted'} "
              f"| {last['iters']} iters")


def record_experiment(meta, r, iters, lr, bs, new_frac, max_forget, source):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "iters": iters,
        "lr": lr,
        "batch": bs,
        "new_frac": new_frac,
        "max_forget": max_forget,
        "reverted": r["reverted"],
        "b_new": r["b_new"],
        "a_new": r["a_new"],
        "b_old": r["b_old"],
        "a_old": r["a_old"],
    }
    meta.setdefault("experiments", []).append(entry)
    if len(meta["experiments"]) > 100:
        meta["experiments"] = meta["experiments"][-100:]


# ----------------------------------------------------------------------------- commands


def cmd_train(a):
    dev = pick_device(a.device)
    preset = pick_auto_preset(dev, a.preset)
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
    ram = available_ram_gb()
    hardware = f"{ram:.1f} GB RAM" if ram else "RAM unknown"
    print(f"device {dev} | preset {preset} | {hardware} | {len(ids):,} bytes "
          f"| {n_params:.1f}M parameters | {iters} iters")

    if os.path.exists(a.ckpt):
        shutil.copyfile(a.ckpt, a.ckpt + ".prev")  # back up once, before this run
    corpus = Corpus(ids, cfg["block"])
    fwd = model
    if a.compile:
        try:
            fwd = torch.compile(model)
        except Exception as e:
            print("torch.compile unavailable, continuing without:", e)

    epochs = iters * bs * cfg["block"] / len(ids)
    print(f"this run shows the model your data ~{epochs:.0f} times"
          + (" - that is a lot, expect memorising; more text would make it smarter." if epochs > 30 else "."))
    base_steps, state = meta.get("steps", 0), {"it": 0}

    def on_best(v):
        meta["steps"] = base_steps + state["it"]  # so interrupted runs still count their progress
        save_ckpt(a.ckpt, model, cfg, meta, backup=False)

    fit(fwd, [corpus], [1.0], iters, lr, bs, dev, warmup=min(100, iters // 10),
        eval_every=a.eval_every, eval_corpus=corpus, on_best=on_best,
        log_every=max(1, min(100, iters // 10)), patience=a.patience, state=state)
    meta["steps"] = base_steps + state["it"]
    ck = torch.load(a.ckpt, map_location=dev, weights_only=True)  # re-save best weights with updated meta
    model.load_state_dict(ck["model"])
    save_ckpt(a.ckpt, model, cfg, meta, backup=False)
    print(f"\nbest checkpoint saved to {a.ckpt}\n--- sample ---\n{sample(model, chr(10), dev)}")


def cmd_gen(a):
    dev = pick_device(a.device)
    model, _, _ = load_ckpt(a.ckpt, dev)
    print(sample(model, a.prompt, dev, a.n, a.temp, a.top_k, a.top_p,
                 a.repetition_penalty, a.frequency_penalty))


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
    old = replay_corpus(try_old_corpus(a, cfg["block"]), meta, cfg["block"])
    if not old:
        print("warning: no replay data or saved lessons - the model may forget what it knew.")
    r = learn(model, encode(text), old, cfg, dev, a.iters, a.lr, a.batch, a.new_frac, a.max_forget, a.force)
    print(report(r, a.max_forget))
    record_experiment(meta, r, a.iters, a.lr, a.batch, a.new_frac, a.max_forget, "learn")
    if r["reverted"]:
        return
    meta["learned"] = meta.get("learned", []) + [(a.files or ["--text"])[0]]
    meta.setdefault("lesson_texts", []).append(text)
    meta["steps"] = meta.get("steps", 0) + a.iters
    save_ckpt(a.ckpt, model, cfg, meta)
    print(f"saved to {a.ckpt} (previous version kept as {a.ckpt}.prev)")


def cmd_chat(a):
    dev = pick_device(a.device)
    model, cfg, meta = load_ckpt(a.ckpt, dev)
    old = replay_corpus(try_old_corpus(a, cfg["block"]), meta, cfg["block"])
    temp, dirty = a.temp, False
    print("Type a prompt and I'll continue it.\n"
          "  /teach <text or @file>   learn something new right now (replays old data to avoid forgetting)\n"
          "  /temp 0.7                change sampling temperature\n"
          "  /lessons                 list saved lessons\n"
          "  /forget N                remove lesson N and replay history\n"
          "  /history                 show learning runs\n"
          "  /status                  show checkpoint statistics\n"
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
            elif cmd == "lessons":
                print_lessons(meta)
            elif cmd == "history":
                print_history(meta)
            elif cmd == "status":
                checkpoint_status(a.ckpt, model, cfg, meta)
            elif cmd == "forget":
                try:
                    index = int(rest)
                except ValueError:
                    print("usage: /forget N")
                    continue
                lessons = meta.get("lesson_texts", [])
                if index < 1 or index > len(lessons):
                    print(f"lesson number must be between 1 and {len(lessons)}")
                    continue
                removed = lessons.pop(index - 1)
                old = replay_corpus(try_old_corpus(a, cfg["block"]), meta, cfg["block"])
                dirty = True
                print(f"removed lesson {index}: {lesson_summary(removed)}")
            elif cmd == "teach" and rest:
                text = clean(open(rest[1:], encoding="utf-8", errors="ignore").read()) if rest.startswith("@") else rest
                if not text.strip():
                    print("Give me some non-empty text to learn.")
                    continue
                r = learn(model, encode(text), old, cfg, dev, iters=a.teach_iters, lr=a.lr,
                          bs=16, new_frac=a.new_frac, max_forget=a.max_forget)
                print(report(r, a.max_forget))
                record_experiment(meta, r, a.teach_iters, a.lr, 16, a.new_frac,
                                  a.max_forget, "chat")
                if not r["reverted"]:
                    meta.setdefault("lesson_texts", []).append(text)
                    if old is None:
                        old = ReplayCorpus(None, [], cfg["block"])
                    old.add(text)
                    dirty = True
            else:
                print("commands: /teach <text|@file>, /temp N, /lessons, /forget N, "
                      "/history, /status, /save, /quit")
        else:
            print(sample(model, line, dev, a.n, temp, a.top_k, a.top_p,
                         a.repetition_penalty, a.frequency_penalty))
    if dirty and input("Save what I learned this session? [y/N] ").strip().lower() == "y":
        save_ckpt(a.ckpt, model, cfg, meta)
        print("saved.")


def cmd_rollback(a):
    prev = a.ckpt + ".prev"
    if not os.path.exists(prev):
        sys.exit("No .prev backup to roll back to.")
    shutil.copyfile(prev, a.ckpt)
    print(f"restored {a.ckpt} from {prev}")


def cmd_lessons(a):
    dev = pick_device(a.device)
    model, cfg, meta = load_ckpt(a.ckpt, dev)
    lessons = meta.setdefault("lesson_texts", [])
    if not a.remove and not a.clear:
        print_lessons(meta)
        return
    if a.clear:
        removed = len(lessons)
        lessons.clear()
        save_ckpt(a.ckpt, model, cfg, meta)
        print(f"removed all {removed} saved lesson(s).")
        return
    if a.remove < 1 or a.remove > len(lessons):
        sys.exit(f"lesson number must be between 1 and {len(lessons)}")
    removed = lessons.pop(a.remove - 1)
    save_ckpt(a.ckpt, model, cfg, meta)
    print(f"removed lesson {a.remove}: {lesson_summary(removed)}")


def cmd_history(a):
    dev = pick_device(a.device)
    _, _, meta = load_ckpt(a.ckpt, dev)
    print_history(meta)


def cmd_status(a):
    dev = pick_device(a.device)
    model, cfg, meta = load_ckpt(a.ckpt, dev)
    checkpoint_status(a.ckpt, model, cfg, meta)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", default=CKPT)
    common.add_argument("--data", default="data")
    common.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    common.add_argument("--seed", type=int, default=1)
    common.add_argument("--threads", type=int, default=0,
                        help="CPU threads (default: leave 4 cores free, up to 8)")
    common.add_argument("--low-priority", action="store_true", help="run at low priority so the PC stays usable")

    gen_args = argparse.ArgumentParser(add_help=False)
    gen_args.add_argument("-n", type=int, default=600, help="bytes to generate")
    gen_args.add_argument("--temp", type=float, default=0.8)
    gen_args.add_argument("--top-k", type=int, default=50)
    gen_args.add_argument("--top-p", type=float, default=0.95)
    gen_args.add_argument("--repetition-penalty", type=float, default=1.15,
                          help="penalise tokens already in context (1 disables)")
    gen_args.add_argument("--frequency-penalty", type=float, default=0.0,
                          help="subtract this per occurrence from token logits")

    learn_args = argparse.ArgumentParser(add_help=False)
    learn_args.add_argument("--lr", type=float, default=2e-4)
    learn_args.add_argument("--new-frac", type=float, default=0.5, help="share of examples from the new text (rest = replay)")
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
    t.add_argument("--patience", type=int, default=8, help="stop after this many validation checks without improvement (0 = never)")
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

    ls = sub.add_parser("lessons", parents=[common], help="list or remove saved lesson replay history")
    lesson_action = ls.add_mutually_exclusive_group()
    lesson_action.add_argument("--remove", type=int, metavar="N", help="remove one saved lesson")
    lesson_action.add_argument("--clear", action="store_true", help="remove all saved lessons")
    ls.set_defaults(fn=cmd_lessons)

    hs = sub.add_parser("history", parents=[common], help="show recent learning runs")
    hs.set_defaults(fn=cmd_history)

    ss = sub.add_parser("status", parents=[common], help="show checkpoint and learning statistics")
    ss.set_defaults(fn=cmd_status)

    a = p.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(a.threads or safe_cpu_threads())
    if a.low_priority:
        lower_priority()
    a.fn(a)


if __name__ == "__main__":
    main()

# tweak 3

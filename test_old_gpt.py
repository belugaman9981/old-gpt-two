import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn


spec = importlib.util.spec_from_file_location("old_gpt", Path(__file__).with_name("old-gpt.py"))
gpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt)


class ConstantCorpus:
    def __init__(self, value):
        self.value = value

    def batch(self, split, bs, dev):
        x = torch.full((bs, 1), self.value, device=dev)
        return x, x


class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(66.0))

    def forward(self, x, y):
        return None, (self.weight - x.float().mean()).square()


class LearningTests(unittest.TestCase):
    def test_repetition_penalty_changes_seen_logits(self):
        logits = torch.tensor([[2.0, -2.0, 1.0]])
        ids = torch.tensor([[0, 1, 1]])
        adjusted = gpt.penalize_repetition(logits, ids, 2.0, 0.5)
        self.assertAlmostEqual(adjusted[0, 0].item(), 0.5)
        self.assertAlmostEqual(adjusted[0, 1].item(), -5.0)
        self.assertAlmostEqual(adjusted[0, 2].item(), 1.0)

    def test_repetition_penalty_rejects_invalid_values(self):
        logits = torch.zeros(1, 2)
        ids = torch.zeros(1, 1, dtype=torch.long)
        with self.assertRaises(ValueError):
            gpt.penalize_repetition(logits, ids, 0)
        with self.assertRaises(ValueError):
            gpt.penalize_repetition(logits, ids, 1, -0.1)

    def test_replay_is_mixed_in_each_batch(self):
        sources = [ConstantCorpus(1), ConstantCorpus(2)]
        x, y = gpt.training_batch(sources, [0.5, 0.5], 8, "cpu")
        self.assertEqual((x == 1).sum().item(), 4)
        self.assertEqual((x == 2).sum().item(), 4)
        self.assertTrue(torch.equal(x, y))
        for fraction in (0.0, 1.0):
            x, _ = gpt.training_batch(sources, [fraction, 1 - fraction], 1, "cpu")
            self.assertEqual(x.item(), 1 if fraction else 2)
        with patch.object(gpt.random, "random", return_value=0.2):
            x, _ = gpt.training_batch(sources, [0.25, 0.75], 1, "cpu")
            self.assertEqual(x.item(), 1)
        with patch.object(gpt.random, "random", return_value=0.3):
            x, _ = gpt.training_batch(sources, [0.25, 0.75], 1, "cpu")
            self.assertEqual(x.item(), 2)

    def test_unchanged_model_uses_same_validation_samples(self):
        model = LossModel()
        old = gpt.Corpus(torch.arange(256), 4, val_frac=0.5)
        with patch.object(gpt, "fit"):
            result = gpt.learn(model, torch.arange(256), old, {"block": 4}, "cpu", bs=2)
        self.assertEqual(result["b_new"], result["a_new"])
        self.assertEqual(result["b_old"], result["a_old"])
        self.assertFalse(result["reverted"])

    def test_forgetting_reverts_weights(self):
        model = LossModel()
        old = gpt.Corpus(torch.full((100,), 66), 4)

        def forget(*args, **kwargs):
            with torch.no_grad():
                model.weight.fill_(0)

        with patch.object(gpt, "fit", side_effect=forget):
            result = gpt.learn(model, torch.full((100,), 65), old, {"block": 4}, "cpu", bs=2)
        self.assertTrue(result["reverted"])
        self.assertEqual(model.weight.item(), 66)

    def test_small_model_really_trains(self):
        torch.manual_seed(1)
        cfg = dict(block=4, n_embd=8, n_head=2, n_layer=1, dropout=0.0)
        model = gpt.GPT(cfg)
        before = model.tok.weight.detach().clone()
        gpt.learn(model, gpt.encode("simple training text"), None, cfg, "cpu", iters=2, bs=2)
        self.assertFalse(torch.equal(before, model.tok.weight))

    def test_replay_balances_original_data_and_lessons(self):
        replay = gpt.ReplayCorpus(ConstantCorpus(90), ["aaaa", "bbbb"], 1)
        with patch.object(gpt.random, "choices", return_value=[0, 1, 0, 1]):
            x, _ = replay.batch("train", 8, "cpu")
        self.assertEqual((x == 90).sum().item(), 4)
        self.assertEqual((x == ord("a")).sum().item(), 2)
        self.assertEqual((x == ord("b")).sum().item(), 2)

    def test_learn_saves_lessons_and_rollback_restores_history(self):
        cfg = dict(block=4, n_embd=8, n_head=2, n_layer=1, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            args = SimpleNamespace(device="cpu", ckpt=path, data=directory, files=[],
                                   text="aaaa", iters=1, lr=0.001, batch=2,
                                   new_frac=0.5, max_forget=0.25, force=False)
            gpt.save_ckpt(path, gpt.GPT(cfg), cfg, {"steps": 0, "learned": []})
            observed = []

            def accept(model, ids, old, *args):
                observed.append(None if old is None else old.batch("train", 2, "cpu")[0])
                return dict(b_new=1.0, a_new=0.5, b_old=None, a_old=None, reverted=False)

            with patch.object(gpt, "learn", side_effect=accept), contextlib.redirect_stdout(io.StringIO()):
                gpt.cmd_learn(args)
                args.text = "bbbb"
                gpt.cmd_learn(args)
            self.assertIsNone(observed[0])
            self.assertTrue(torch.all(observed[1] == ord("a")))
            self.assertEqual(gpt.load_ckpt(path, "cpu")[2]["lesson_texts"], ["aaaa", "bbbb"])
            with contextlib.redirect_stdout(io.StringIO()):
                gpt.cmd_rollback(args)
            self.assertEqual(gpt.load_ckpt(path, "cpu")[2]["lesson_texts"], ["aaaa"])

    def test_chat_replays_accepted_lessons_and_only_saves_on_request(self):
        cfg = dict(block=4, n_embd=8, n_head=2, n_layer=1, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            gpt.save_ckpt(path, gpt.GPT(cfg), cfg, {"steps": 0, "learned": []})
            args = SimpleNamespace(device="cpu", ckpt=path, data=directory, temp=0.8,
                                   teach_iters=1, lr=0.001, new_frac=0.5,
                                   max_forget=0.25, n=1, top_k=1, top_p=1.0)
            observed = []

            def teach(model, ids, old, *args, **kwargs):
                observed.append(None if old is None else old.batch("train", 2, "cpu")[0])
                return dict(b_new=1.0, a_new=0.5, b_old=None, a_old=None,
                            reverted=len(observed) == 2)

            inputs = ["/teach aaaa", "/teach bbbb", "/save", "/teach cccc", "/quit", "n"]
            with patch.object(gpt, "learn", side_effect=teach), \
                    patch("builtins.input", side_effect=inputs), contextlib.redirect_stdout(io.StringIO()):
                gpt.cmd_chat(args)
            self.assertIsNone(observed[0])
            self.assertTrue(torch.all(observed[1] == ord("a")))
            self.assertTrue(torch.all(observed[2] == ord("a")))
            self.assertEqual(gpt.load_ckpt(path, "cpu")[2]["lesson_texts"], ["aaaa"])

    def test_two_real_learning_runs_replay_saved_lesson(self):
        torch.manual_seed(1)
        cfg = dict(block=4, n_embd=8, n_head=2, n_layer=1, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            gpt.save_ckpt(path, gpt.GPT(cfg), cfg, {"steps": 0, "learned": []})
            args = SimpleNamespace(device="cpu", ckpt=path, data=directory, files=[],
                                   text="first lesson", iters=2, lr=0.001, batch=2,
                                   new_frac=0.5, max_forget=0.25, force=True)
            with contextlib.redirect_stdout(io.StringIO()):
                gpt.cmd_learn(args)
                args.text = "second lesson"
                gpt.cmd_learn(args)
            self.assertEqual(gpt.load_ckpt(path, "cpu")[2]["lesson_texts"],
                             ["first lesson", "second lesson"])

    def test_lesson_management_and_experiment_history(self):
        cfg = dict(block=4, n_embd=8, n_head=2, n_layer=1, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            meta = {
                "steps": 0,
                "learned": [],
                "lesson_texts": ["first lesson", "second lesson"],
            }
            gpt.save_ckpt(path, gpt.GPT(cfg), cfg, meta)
            result = dict(b_new=1.0, a_new=0.8, b_old=1.1, a_old=1.2,
                          reverted=False)
            gpt.record_experiment(meta, result, 100, 0.0002, 16, 0.5, 0.25, "learn")
            gpt.save_ckpt(path, gpt.GPT(cfg), cfg, meta)

            remove_args = SimpleNamespace(device="cpu", ckpt=path, remove=1, clear=False)
            with contextlib.redirect_stdout(io.StringIO()):
                gpt.cmd_lessons(remove_args)
            saved_meta = gpt.load_ckpt(path, "cpu")[2]
            self.assertEqual(saved_meta["lesson_texts"], ["second lesson"])

            history_output = io.StringIO()
            with contextlib.redirect_stdout(history_output):
                gpt.print_history(saved_meta)
            self.assertIn("accepted", history_output.getvalue())
            self.assertIn("1 learning run", history_output.getvalue())

    def test_experiment_history_is_capped(self):
        meta = {"experiments": []}
        result = dict(b_new=1.0, a_new=0.8, b_old=None, a_old=None,
                      reverted=False)
        for i in range(105):
            gpt.record_experiment(meta, result, i, 0.0002, 16, 0.5, 0.25, "learn")
        self.assertEqual(len(meta["experiments"]), 100)
        self.assertEqual(meta["experiments"][0]["iters"], 5)

    def test_status_reports_checkpoint_and_learning_totals(self):
        cfg = dict(block=4, n_embd=8, n_head=2, n_layer=1, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            meta = {
                "steps": 42,
                "lesson_texts": ["one", "two"],
                "experiments": [
                    {"timestamp": "2026-01-01 00:00:00", "reverted": False, "iters": 10},
                    {"timestamp": "2026-01-02 00:00:00", "reverted": True, "iters": 20},
                ],
            }
            gpt.save_ckpt(path, gpt.GPT(cfg), cfg, meta)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                gpt.checkpoint_status(path, gpt.GPT(cfg), cfg, meta)
            text = output.getvalue()
            self.assertIn("42", text)
            self.assertIn("saved lessons: 2", text)
            self.assertIn("2 runs", text)
            self.assertIn("1 accepted", text)
            self.assertIn("1 reverted", text)

    def test_cpu_auto_preset_uses_available_hardware(self):
        with patch.object(gpt, "available_ram_gb", return_value=32.0), \
                patch.object(gpt.os, "cpu_count", return_value=16):
            self.assertEqual(gpt.pick_auto_preset("cpu", "auto"), "base")
            self.assertEqual(gpt.pick_auto_preset("cpu", "tiny"), "tiny")
            self.assertEqual(gpt.safe_cpu_threads(), 8)
        with patch.object(gpt, "available_ram_gb", return_value=8.0), \
                patch.object(gpt.os, "cpu_count", return_value=8):
            self.assertEqual(gpt.pick_auto_preset("cpu", "auto"), "small")
        with patch.object(gpt, "available_ram_gb", return_value=4.0), \
                patch.object(gpt.os, "cpu_count", return_value=4):
            self.assertEqual(gpt.pick_auto_preset("cpu", "auto"), "tiny")
            self.assertEqual(gpt.safe_cpu_threads(), 4)


if __name__ == "__main__":
    unittest.main()
 

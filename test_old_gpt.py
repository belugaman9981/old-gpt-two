import importlib.util
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()

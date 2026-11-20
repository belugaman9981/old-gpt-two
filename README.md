# old-gpt-two

Training uses at most four CPU threads by default to keep the computer responsive.
Use `--threads N` to choose a different limit, and `--low-priority` to reduce
the process priority. Training and learning show smoothed loss and an estimated
remaining time between validation reports.

`learn` and chat's `/teach` mix new and replay examples in each batch when
replay data is available. `--new-frac` controls the share of new examples.
Forgetting is checked using the same sampled validation batches before and
after learning; the model is reverted if old-data loss rises above
`--max-forget` unless `learn --force` is used. Small texts may be repeated to
make batches, so their validation loss is not necessarily a held-out measure.

Accepted `learn` and `/teach` text is stored in full in the checkpoint and
replayed during later `learn` or `/teach` sessions. If both original data and
lessons are available, each receives half of the replay examples; lessons are
sampled equally. A lesson rejected by the forgetting check is not saved.
Chat lessons are available immediately during the session, but only persist
if you save. Checkpoint backups and rollback include the lesson history.
Older checkpoints do not contain lesson text, so past lessons cannot be
replayed unless taught again. Treat checkpoints and their backups as containing
your lesson text. `train --resume` still trains on the data directory only.

Use `python old-gpt.py lessons` to list saved lessons. Remove one with
`python old-gpt.py lessons --remove N`, or clear all with
`python old-gpt.py lessons --clear`. In chat, use `/lessons`, `/forget N`, and
`/history`. The most recent 100 learning runs are retained in checkpoint
metadata; they include settings and before/after loss values.

Use `python old-gpt.py status` for checkpoint size, model shape, training steps,
saved lesson count, and learning-run totals. Chat also supports `/status`.

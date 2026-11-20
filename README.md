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

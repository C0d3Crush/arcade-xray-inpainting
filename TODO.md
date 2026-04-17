# TODO

## Immediate
- [ ] Push commits to GitLab (`git push gitlab`) — fixes broken Auto DevOps pipeline
- [ ] Verify Colab training converges (check `checkpoints/training_log.csv` after ~10 epochs)

## Training
- [ ] Tune `--n_shapes` in mask generation (currently 5) — more shapes = harder task
- [ ] Try `--input_size 64` for fast iteration runs before committing to full 256 runs
- [ ] Experiment with batch size (currently 16 on T4 — may OOM, fall back to 8 if needed)

## Evaluation
- [ ] Add inference visualization cell to notebook (overlay inpainted result on original)
- [ ] Evaluate best checkpoint on held-out test images with `demo.py`

## GitLab CI
- [ ] Verify `smoke_test` job passes after pushing `.gitlab-ci.yml`

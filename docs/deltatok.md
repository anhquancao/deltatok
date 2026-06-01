# DeltaTok

## Eval-only

```bash
EVAL_ONLY=1 bash sh/train_deltatok.sh
```

Loads the checkpoint from `<LOG_AND_CKPT_DIR>/<RUN_NAME>/ckpts/` and writes
visualisation panels to `<RESULTS_DIR>/<RUN_NAME>/eval_only/eval_depth/<test_name>/`.

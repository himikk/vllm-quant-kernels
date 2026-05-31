# Test fixtures

Real model tensors used by `tests/test_real_lm_head.py` for end-to-end accuracy
validation against `lm_head` weights from production models.

## Files

| File | Source | Size |
|------|--------|------|
| `qwen3p5_122b_lm_head.safetensors` | `Intel/Qwen3.5-122B-A10B-int4-AutoRound`, shard 14, `lm_head.weight` (bf16) | 1.5 GB |

These files are **not committed to git** (see `.gitignore`). Tests that need
them skip gracefully when the file is missing.

## Regenerating

If you have the source model cached locally in `~/.cache/huggingface/hub`,
extract the fixture by running:

```bash
python tests/fixtures/extract_lm_head.py
```

The script looks for the model under
`~/.cache/huggingface/hub/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/`.
Set `HF_HUB` to override the cache root.

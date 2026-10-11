# qwen35 (Bonsai) spec pin

Treat the PrismML llama.cpp fork as the architecture spec:

- Repo: `PrismML-Eng/llama.cpp`
- Branch: `prism`
- File: `src/models/qwen35.cpp`
- Commit SHA: **not vendored in this tree** (pin when the fork is imported). Until then, the layer-kind rule below is the portable subset of lines 8–30.

## Layer kind (qwen35.cpp ~L8–L30)

The fork marks recurrent (gated-delta / linear-attention) layers on load.

1. If GGUF `{arch}.attention.layer_types` is present:
   - string values `linear_attention` / `recurrent` / `gated_delta` / `delta_net` / `ssm` → recurrent
   - `full_attention` / `attention` / `full` / `mha` → full softmax attention
   - integer array: `0` = full, nonzero = recurrent
2. Else the default 3+1 hybrid (~75% linear, ~25% full):
   - layer `il` is **full** iff `(il + 1) % 4 == 0`
   - otherwise **recurrent** (gated delta net)

## Ops

| Kind | Path |
|---|---|
| Full attention | RMSNorm, Q/K/V, per-head RMSNorm, `rope_multi` + `RopeParams.sections`, `Backend::attention` |
| Linear attention | RMSNorm, mixed QKV, causal `ssm_conv` (`conv1d_causal`), `l2_norm` on q/k, softplus/sigmoid gates, `Backend::gated_delta_step` |
| Both | SwiGLU FFN, RMSNorm |

MTP / next-n tensors (`mtp.*`, `nextn.*`, `blk.{i}` with `i >= n_layer`) are **loaded and not run**.

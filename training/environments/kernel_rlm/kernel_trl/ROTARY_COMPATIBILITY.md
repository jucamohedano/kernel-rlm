# Laguna rotary compatibility

## Observed failure

- The checkpoint specifies 64 rotary dimensions in global attention and 128 in
  sliding attention (`head_dim=128`, fractions 0.5 and 1.0).
- It also retains a legacy top-level `partial_rotary_factor=0.5`.
- Native Transformers 5.16.1 constructs only 64 dimensions for sliding attention.
  vLLM 0.25.1 constructs 128. We measured the actual loaded objects, not only JSON.
- Global attention outputs agreed closely. The first sliding-attention layer
  diverged before the first MoE block. Window-size and norm-rounding variants
  did not resolve the discrepancy.

The old CPU cache comparison was misleading: constructing the Transformers
configuration mutated the nested dictionary that was then passed to the other
implementation. Both test objects therefore used the same altered settings.
New checks preserve the original checkpoint dictionary independently.

## Loader correction

`kernel_training.laguna_policy.load_config` reads the original per-layer-type settings
after automatic config parsing, restores an independent copy, and removes real
legacy instance fields. It does not delete read-only compatibility properties.
Removing the legacy fields matters: restoring the dictionary alone is undone
when model construction normalizes the config again.

Both rank zero's pretrained loader and rank one's meta constructor use this
helper. It does not repack or change any pretrained weights, increase context,
alter token IDs or replace behavior probabilities. CPU construction now verifies
frequency-buffer lengths 32 and 64: two coordinates per frequency give the required
64 and 128 rotary dimensions.

## Validation gate

Run the full unchanged-policy probability audit after this correction. Keep the
original failed audit and any aborted attempts separate. A CPU geometry proof is
not enough to claim sampler/learner agreement, safe GPU memory or improved model
quality. Training remains gated on a passing exact-group audit.

The corrected two-rank audit scored all 10,631 saved root target tokens with zero
updates. Mean absolute log-ratio fell from 5.1716 to 0.02225; the fraction outside
ratios 0.8–1.2 fell from 93.25% to 3.217%. Both unchanged readiness limits pass.
The maximum token discrepancy is still 1.656; this is numerical agreement within
the measured gate, not bitwise equivalence or proof of learning.

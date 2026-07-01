# HFM-1D — Latent-Rollout Fluid Model

A next-frame fluid predictor that evolves a compact **latent state** forward in
time and decodes it to images, rather than re-encoding pixels every step. It is
the neural analogue of an FVM solver: a small state advanced by a learned
integrator, rendered to observables periodically.

## Idea

```
x0 ──shallow skip encoder──────────────► skip_feats   (initial-frame detail anchor)
x0 ──frame encoder──► slots S0
                        │
          ┌─────────────┴──────────── for i = 0 .. horizon-1 ────────────┐
          │  Sᵢ₊₁ = Evolve(Sᵢ, context, i)     (weight-shared integrator) │
          │  x̂ᵢ₊₁ = Decode(Sᵢ₊₁, skip_feats)                             │
          │  (optionally every m steps: S ← Encode(x̂), refresh skip)     │
          └───────────────────────────────────────────────────────────────┘
```

Design decisions (see the derivation in the project notes):

- **Pure slot bottleneck.** The encoder distills a frame into `n_slots` tokens
  (Perceiver-style). Only the slots are rolled forward — the evolution operator
  never sees a full patch grid. This is what makes latent rollout cheaper than
  re-predicting pixels every step.
- **Skip anchor from the *input*, not the encoder output.** Fine spatial detail
  reaches the decoder through a shallow conv on the raw initial frame, kept
  separate from the deep encoder features. Routing encoder *outputs* into the
  decoder would let it reconstruct without the slots and collapse the
  bottleneck; shallow input features preserve detail without that shortcut.
- **Weight-shared, tendency-recomputing integrator.** One `EvolutionOperator`
  defines `dS/dτ = f(S, context)` and integrates with Euler or midpoint (RK2),
  recomputing the tendency at each stage. Shared weights give a genuine
  integrator with a variable-horizon / variable-Δt story. The tendency head is
  zero-initialised, so training starts from the identity map.
- **Supervise every step + periodic re-encode.** The rollout is decoded and
  supervised at every horizon step (not just the end), and `reencode_every`
  re-anchors the latent to a fresh encode to bound drift.
- **Context conditioning.** A `ContextEncoder` summarises the history frames
  into `K` tokens that tell the integrator *which* dynamics to advance.

## Layout

```
hfm1d/
  config.py           HFM1DConfig
  attention.py        multi-head cross-attention
  modules.py          FeedForward, PatchEmbed, LearnedPos2D, SkipEncoder, blocks
  encoder.py          FrameEncoder     (frame → slots)
  evolution.py        EvolutionOperator (weight-shared latent integrator)
  decoder.py          SlotDecoder      (slots → patch grid → image, tent fold)
  context_encoder.py  ContextEncoder   (history → K context tokens)
  discriminator.py    HFMDiscriminator (PatchGAN-style, context-conditioned)
  model.py            HFM1D            (encode → rollout → decode)
  trainer.py          RolloutGANTrainer + train_step_gan + FluidLoss
  data.py             FVM dataset / renderer pipeline
scripts/train.py      training entry point
hyperparams.json      model + training hyperparameters
```

## Training

```bash
cd hfm1d
python scripts/train.py --data /path/to/fvm_dataset
```

Curriculum: reconstruction-only until step 10k, then the adversarial weight
ramps in over 2k steps. The discriminator only updates while its loss is in
`(0.5, 2.0)`.

## Smoke test

```bash
python scripts/smoke_test.py
```

Builds a tiny model and runs a forward + backward pass on random data — no
dataset required.

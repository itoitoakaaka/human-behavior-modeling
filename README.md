# Human Behavior Modeling

A compact portfolio project for modeling repeated human behavior with classical statistics and PyTorch.

## Goal

The same trial-by-trial behavioral sequence is modeled with several levels of complexity:

1. naive persistence baseline
2. linear regression
3. multilayer perceptron (MLP)
4. gated recurrent unit (GRU)

The point is not to make deep learning win. The point is to compare transparent baselines with nonlinear and sequence-based models under the same evaluation scheme.

## Synthetic task

The demo creates participant-level adaptation sequences with:

- target
- previous error
- previous output
- condition
- experience group

The prediction target is the next-trial behavioral output.

All data are synthetic.

## Evaluation

Participants are split into train/test sets so trials from the same person do not leak across evaluation sets.

Metrics:

- MAE
- RMSE
- R2

## Files

- `data.py`: synthetic participant-level behavior generator
- `features.py`: supervised-learning dataset construction
- `baselines.py`: persistence and linear-regression baselines
- `torch_models.py`: MLP and GRU models
- `evaluate.py`: shared metrics
- `demo.py`: end-to-end comparison
- `tests/test_pipeline.py`: lightweight tests

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run

    python demo.py

## Research direction

A natural next step is to replace the synthetic generator with real trial-by-trial behavioral data and ask whether computational models capture individual differences across environments.

This repository complements `computational-sensorimotor-modeling`:

- that repository emphasizes interpretable latent-state models
- this repository emphasizes predictive human-behavior modeling

Together they provide a bridge from experimental human data to computational neuroscience and Physical AI.

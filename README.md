# Human Behavior Modeling

[![tests](https://github.com/itoitoakaaka/human-behavior-modeling/actions/workflows/tests.yml/badge.svg)](https://github.com/itoitoakaaka/human-behavior-modeling/actions/workflows/tests.yml)

Leakage-aware predictive modeling of trial-by-trial human behavior with transparent baselines and PyTorch sequence models.

## Question

Can recent trial history predict the next behavioral output, and does a nonlinear or sequence model add value beyond simple baselines?

The public demo is synthetic. Its purpose is to make the evaluation logic explicit before applying it to real participant data.

## Models

1. persistence baseline
2. standardized ridge regression
3. PyTorch multilayer perceptron
4. PyTorch GRU

Deep learning is not assumed to be better. Every model is evaluated against the same held-out participants.

## Leakage control

Trials from one participant are never split across train and test sets.

Training, validation, and test participants are separated before model fitting. Feature standardization is estimated from training data only.

That matters more than decorating the README with sixteen badges and hoping nobody notices the split strategy.

## Prediction task

Inputs include:

- current target
- current output
- current error
- condition indicator
- experience-group indicator
- recent history for the GRU model

Target:

- next-trial behavioral output

## Install

```bash
git clone https://github.com/itoitoakaaka/human-behavior-modeling.git
cd human-behavior-modeling
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
human-behavior-demo
```

Run tests:

```bash
pytest
```

## Evaluation

Reported metrics:

- MAE
- RMSE
- R²

The demo uses a participant-level train/validation/test split.

## Repository layout

```text
src/behavior_modeling/
  data.py
  features.py
  baselines.py
  torch_models.py
  evaluate.py
  demo.py
tests/
.github/workflows/tests.yml
pyproject.toml
```

## Relationship to computational-sensorimotor-modeling

This repository focuses on **prediction**.

[`computational-sensorimotor-modeling`](https://github.com/itoitoakaaka/computational-sensorimotor-modeling) focuses on **interpretable latent-state parameters** such as retention and error sensitivity.

The useful comparison is therefore not "classical vs deep learning" as a team sport. It is:

- what an interpretable mechanistic model explains
- what a predictive model forecasts
- whether added model complexity generalizes to unseen participants


## Decision modeling, intervention, and HCI

<p align="center">
  <img src="assets/decision_hci_overview.svg" alt="Decision modeling and human-centered intervention pipeline" width="100%">
</p>


The repository now also includes a compact decision-science extension:

- quasi-hyperbolic (beta-delta) temporal discounting
- softmax choice probabilities
- synthetic intervention policies with explicit user-burden costs
- HCI evaluation helpers that keep behavioral effectiveness separate from usefulness, cognitive load, and autonomy cost
- a small research note connecting decision models to adaptive human-centered intervention

Run:

```bash
python examples/decision_hci_demo.py
```

See `RESEARCH_NOTE.md` for the conceptual framing.

The long-term research direction is:

```text
human sensing
→ latent-state / decision modeling
→ behavioral prediction
→ adaptive intervention
→ human response
```

The new examples are synthetic and educational. They are not presented as validated models of a specific population or as evidence for intervention effectiveness.

## Current limitation

All public results are synthetic. No human-study finding is claimed here.

The next research step is a preregistered or otherwise clearly specified evaluation on de-identified real trial-by-trial data, with participant-level cross-validation and model comparison.


## Adaptive algorithmic recourse prototype

<p align="center">
  <img src="assets/adaptive_recourse_overview.svg" alt="Adaptive algorithmic recourse under latent user state" width="100%">
</p>

This repository also contains a **synthetic extension inspired by** Tominaga, Yamashita, and Kurashima, *Psychological Benefits and Costs of Diversifying Algorithmic Recourse* (IJCAI-ECAI 2026).

A paper-style technical note suitable for archiving with a software release is available at [`ADAPTIVE_RECOURSE_TECHNICAL_NOTE.md`](ADAPTIVE_RECOURSE_TECHNICAL_NOTE.md).

The published study reports a trade-off between psychological benefits and cognitive costs of recourse-set size/diversity. The implementation here does **not** reproduce their participant data or estimate their empirical effect sizes. Instead, it asks a prospective question:

> Can the number and diversity of recourse options be adapted to a user's latent engagement, cognitive-load, and acceptance state?

The prototype contains:

- a transparent latent user state: engagement, cognitive load, acceptance
- candidate recourse policies: 1, 3, or 7 options, with close/diverse variants where meaningful
- an explicit utility balancing willingness to act and decision acceptance against cognitive load
- state-dependent policy selection
- a trial-to-trial state update for closed-loop simulation

Run:

```bash
python examples/adaptive_recourse_demo.py
```

Implementation:

```text
src/behavior_modeling/adaptive_recourse.py
examples/adaptive_recourse_demo.py
tests/test_adaptive_recourse.py
```

### Why this is an extension rather than a reproduction

The original authors' public repository provides analysis code and describes the expected experimental dataset, but the participant-level dataset is not publicly bundled and may be available only on reasonable request. This project therefore keeps empirical reproduction and speculative modeling separate.

The coefficients in `adaptive_recourse.py` are illustrative. They encode only the published qualitative pattern: diversity can be beneficial for smaller sets, while a large diverse set can make cognitive load more salient.

A real next step would estimate the state and response model from human data, compare adaptive vs fixed policies prospectively, and evaluate effectiveness, cognitive load, autonomy, and acceptance separately.

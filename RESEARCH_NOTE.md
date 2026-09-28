# Research Note: From Decision Models to Human-Centered Intervention

## Motivation

A useful model of human behavior should do more than predict what a person will do.
For human-centered systems, the model can also be used to ask:

1. What latent preference or bias could explain the observed choice?
2. Which intervention might change the behavior?
3. What cost does that intervention impose on the user?
4. Does the intervention remain useful when individual differences and uncertainty are considered?

This repository adds a small decision-modeling layer to the existing trial-by-trial
behavior-prediction code.

## 1. Intertemporal choice

The first model is quasi-hyperbolic (beta-delta) discounting:

```text
D(0) = 1
D(t) = beta * delta^t,  t > 0
```

where:

- `beta` represents present bias,
- `delta` represents longer-term discounting.

Choice is converted to a probability with a softmax rule. This is deliberately simple:
the point is to expose assumptions rather than hide them inside a large predictive model.

## 2. Intervention simulation

A second module treats an intervention as having both:

- a potential behavioral benefit,
- a user burden.

Synthetic policies vary in prompt strength and burden. The example asks whether stronger
intervention is actually better once burden is penalized.

That distinction is important for HCI. Maximizing behavioral compliance alone is not the
same as designing a good human-centered system.

## 3. HCI evaluation

The HCI helper keeps behavioral effectiveness separate from:

- perceived usefulness,
- cognitive load,
- autonomy cost.

The composite score is only a convenience for demos. In a real study, these outcomes
should be reported separately and the weighting should be preregistered or justified.

## 4. Connection to my research direction

My current research background is in human sensorimotor control and trial-by-trial
behavior. The next conceptual step is to connect:

```text
human sensing
→ latent-state / decision modeling
→ behavioral prediction
→ adaptive intervention
→ human response
```

This creates a bridge between computational neuroscience, decision science, and
human-centered AI.

## 5. What this note does not claim

- The synthetic intervention is not a validated behavioral intervention.
- The beta-delta model is not assumed to explain all human decisions.
- The HCI score is not a universal metric.
- No human-study result is claimed from the synthetic examples.

## Next steps

1. reproduce a published intervention / decision-modeling result,
2. add parameter recovery for beta and delta,
3. add partial observability / latent-state uncertainty,
4. compare static and adaptive intervention policies,
5. evaluate intervention effectiveness and subjective burden in a human study.

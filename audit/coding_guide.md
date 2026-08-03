# Literature Audit Coding Guide

## Inclusion Criterion

A paper is 'empirical' iff it applies an ML method to real cryptocurrency
market data for a trading or prediction task. Excluded: surveys,
synthetic-only methodological studies, non-crypto asset studies,
non-ML studies. Excluded ids: 1, 9, 11, 12, 71, 79.

## Dimensions

| Code | Question | Values |
|------|----------|--------|
| D1 | Are transaction costs included? | Yes / No / Partial |
| D1_bp | Cost assumption (basis points) | integer or "not specified" |
| D2 | Is class balance addressed? | Yes / No / Partial / N/A |
| D3 | Temporal train/test split? | Yes temporal / No random |
| D4 | Validation method? | CPCV / Walk-forward / k-fold / Holdout |
| D5 | Baselines compared? | BnH / BnH+rules / ML only / None |
| D6 | Net performance reported? | Yes / No / Partial |
| D7 | Code available? | Yes GitHub / Yes other / No |

## Coding Rules

- When ambiguous, use the most generous interpretation
- "Partial" = mentioned but not clearly applied
- D1_bp = 0 if costs are explicitly omitted
- D4: Random k-fold on time series = "k-fold" (flagged as weak)
- D5: "BnH+rules" requires at least one technical indicator baseline

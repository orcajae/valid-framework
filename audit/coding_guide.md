# Literature Audit Coding Guide

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

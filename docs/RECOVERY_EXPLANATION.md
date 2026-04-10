# SmartGrid recovery pass

## What was done
- Kept `src/Legacy/legacy_conso_local_torch.py` as the historical reference.
- Canonized `src/smartgrid/` as the only package to extend.
- Rebuilt the consumption pipeline into modules:
  - loading
  - feature engineering
  - temporal split
  - PyTorch MLP model
  - training loop
  - notebook-aligned metrics
  - artifact publication / promotion
  - simple API bound to the promoted model

## What was intentionally *not* done yet
- production PV pipeline
- live collector/scheduler
- front-end
- battery control
- advanced orchestration

## Why this is the right first pass
Because the urgent need is not a perfect platform.
The urgent need is a clean training path that can produce comparable results fast without breaking inference.

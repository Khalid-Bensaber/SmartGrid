# Next steps after the recovery pass

## 1. What is now ready
- A canonical package path: `src/smartgrid/`
- A modular consumption pipeline
- A PyTorch MLP training entrypoint
- Notebook-aligned evaluation and exports
- A simple model registry with `current`
- A simple API that reads the promoted model only

## 2. What you should do next
1. Run a first real training pass on consumption.
2. Check the produced `artifacts/exports/consumption/<run_id>/` files.
3. Compare the generated export in the original notebook.
4. Add the first experiment variants:
   - fewer lag days
   - more lag days
   - different hidden layers
   - dropout vs no dropout
5. Once the consumption pipeline is stable, clone the same architecture for PV production.

## 3. Recommended immediate experiment matrix
- `lag_days = [7,1,2,3,4,5,6]`, hidden `[1024,512]`
- `lag_days = [7,1,2,3]`, hidden `[512,256]`
- `lag_days = [7,1]`, hidden `[256,128]`
- `lag_days = [7,1,2,3,4,5,6]`, hidden `[1024,512]`, dropout `0.1`

## 4. Strict rule from now on
Do not add new forecasting code in:
- `src/smart_grid/`
- `src/smartgrid/api.py` old flat style
- `src/smartgrid/service.py` old flat style

All new work must go into the modular folders.

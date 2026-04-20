from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def predict_from_feature_dict(model, x_scaler, y_scaler, features: dict[str, float], feature_columns: list[str], device: torch.device) -> float:
    row = pd.DataFrame([{column: features[column] for column in feature_columns}])
    x_scaled = x_scaler.transform(row.to_numpy(dtype=float))
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()
    prediction = y_scaler.inverse_transform(pred_scaled).ravel()[0]
    return float(prediction)


def predict_from_feature_matrix(
    model,
    x_scaler,
    y_scaler,
    feature_matrix: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_scaled = x_scaler.transform(np.asarray(feature_matrix, dtype=float))
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_scaled = model(x_tensor).detach().cpu().numpy()
    return y_scaler.inverse_transform(pred_scaled).ravel()

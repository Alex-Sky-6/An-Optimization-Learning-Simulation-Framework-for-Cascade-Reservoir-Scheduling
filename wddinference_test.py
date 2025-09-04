# inference_api.py
import torch
import numpy as np
import pandas as pd
import joblib

# === TODO(1): Change to your training script filename & model class name ===
from wddlearning import LSTMPredictor as ModelClass  

# === TODO(2): Fill in according to training configuration ===
SEQ_LEN = 3                             # Window length during training
INPUT_FEATURES = ['Wudongde_Inflow','Wubai_Inflow'] # Input columns used during training, order must be consistent!
MODEL_KW = dict(
    input_size=len(INPUT_FEATURES),     # Must be consistent with training
    hidden_size=128,
    num_layers=2,
    output_size=1,
    dropout=0.2,
)

MODEL_PATH = "model_best.pt"            # Model weights saved in previous step
SCALER_INPUT_PATH = "scaler_input.pkl"  # Scaler saved in previous step
SCALER_TARGET_PATH = "scaler_target.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelAPI:
    def __init__(self):
        # 1) Build model with same structure and load weights
        self.model = ModelClass(**MODEL_KW).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        # 2) Load scaler
        self.scaler_input = joblib.load(SCALER_INPUT_PATH)
        self.scaler_target = joblib.load(SCALER_TARGET_PATH)

    def __call__(self, df: pd.DataFrame) -> float:
        """Use like a function: return next step prediction (original scale, scalar)"""
        return self.predict_next_step(df)

    @torch.no_grad()
    def predict_next_step(self, df: pd.DataFrame) -> float:
        """Use the most recent SEQ_LEN records to predict next step (single step)"""
        vals = df[INPUT_FEATURES].values
        if len(vals) < SEQ_LEN:
            raise ValueError(f"Insufficient data length: need at least {SEQ_LEN} rows containing {INPUT_FEATURES} data")

        X = self.scaler_input.transform(vals)   # Input normalization consistent with training
        window = X[-SEQ_LEN:]                   # [T, C]
        x = torch.from_numpy(window).float().unsqueeze(0).to(DEVICE)  # [1,T,C]

        y_norm = self.model(x).cpu().numpy()    # [1,1]
        y = self.scaler_target.inverse_transform(y_norm)  # Inverse normalize back to original scale
        return float(y.squeeze())

    @torch.no_grad()
    def rolling_predict(self, df: pd.DataFrame, steps: int = 12):
        """Rolling prediction for future N steps (common scenario: target column is also in input features, e.g., outflow)"""
        hist = self.scaler_input.transform(df[INPUT_FEATURES].values)
        if len(hist) < SEQ_LEN:
            raise ValueError(f"Insufficient data length: need at least {SEQ_LEN} rows")

        history = hist.copy()
        outputs = []
        target_col = self._guess_target()

        for _ in range(steps):
            window = history[-SEQ_LEN:]
            x = torch.from_numpy(window).float().unsqueeze(0).to(DEVICE)
            y_norm = self.model(x).cpu().numpy()
            y = self.scaler_target.inverse_transform(y_norm)  # Original scale
            outputs.append(float(y.squeeze()))

            # Write predicted target value back to "next moment" input (if target column is in input features)
            last_denorm = df[INPUT_FEATURES].iloc[-1].to_dict()
            if target_col in last_denorm:
                last_denorm[target_col] = outputs[-1]
            next_row_denorm = np.array([[last_denorm[c] for c in INPUT_FEATURES]], dtype=float)
            next_row_norm = self.scaler_input.transform(next_row_denorm)
            history = np.vstack([history, next_row_norm])

        return outputs

    def _guess_target(self):
        return "outflow" if "outflow" in INPUT_FEATURES else INPUT_FEATURES[-1]

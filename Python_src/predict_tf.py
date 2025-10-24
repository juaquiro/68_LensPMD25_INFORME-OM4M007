
"""
# Inference helper for the training results of Train_TF_Regression_Multioutput
Load trained model, scaler, and metadata to make predictions on new data.

68 LENSPMD25 INFORME-OM4M007 medida spatial freqs using ML

"""


from pathlib import Path
import pickle
import json
import pandas as pd
from tensorflow import keras

def load_artifacts(model_dir: str | Path,
                   scaler_pkl: str | Path,
                   meta_json: str | Path):
    model = keras.models.load_model(Path(model_dir).as_posix())
    with open(scaler_pkl, 'rb') as f:
        scaler = pickle.load(f)
    with open(meta_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, scaler, meta

def predict_from_dataframe(df: pd.DataFrame,
                           model_dir: str | Path,
                           scaler_pkl: str | Path,
                           meta_json: str | Path) -> pd.DataFrame:
    model, scaler, meta = load_artifacts(model_dir, scaler_pkl, meta_json)
    req_vars = meta['predictorNameList']
    missing = [v for v in req_vars if v not in df.columns]
    if missing:
        raise ValueError(f'Missing required predictor(s): {missing}')
    X = df[req_vars].values
    X_std = scaler.transform(X)
    y_pred = model.predict(X_std)
    return pd.DataFrame(y_pred, columns=meta['responseNameList'])

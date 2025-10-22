"""
Streamlit app for Customer Churn Prediction

Features:
- Load a pickled model (supports dicts containing 'model','scaler','feature_names') or upload a model file in the UI
- Robust discovery of expected feature names (from saved metadata, scaler, model, or xgboost booster)
- Align incoming input (single-record via form or batch via CSV) to training features (missing -> 0)
- Scaling using saved scaler when available
- Predict with probabilities when available
- Downloadable CSV with predictions

Run:
    pip install streamlit pandas scikit-learn xgboost
    streamlit run app.py

"""

from __future__ import annotations
import io
import os
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_streamlit")

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")


@st.cache_resource
def load_model_from_path(path: str) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """Load a pickled model file from disk.

    Returns (model, scaler, feature_names)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    with open(path, "rb") as f:
        loaded = pickle.load(f)

    return _parse_loaded_object(loaded)


@st.cache_resource
def load_model_from_bytes(b: bytes) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """Load a pickled model from bytes (e.g. uploaded file)."""
    loaded = pickle.loads(b)
    return _parse_loaded_object(loaded)


def _parse_loaded_object(loaded: Any) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    model = None
    scaler = None
    feature_names = None

    if isinstance(loaded, dict):
        model = loaded.get("model") or loaded.get("clf") or loaded.get("estimator")
        scaler = loaded.get("scaler") or loaded.get("preprocessor")
        feature_names = loaded.get("feature_names") or loaded.get("columns")

    if model is None and isinstance(loaded, (list, tuple)) and len(loaded) >= 1:
        model = loaded[0]
        if len(loaded) >= 2:
            scaler = loaded[1]
        if len(loaded) >= 3:
            feature_names = loaded[2]

    if model is None:
        model = loaded

    # discover feature names
    if feature_names is None:
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            try:
                feature_names = list(getattr(scaler, "feature_names_in_") )
                logger.info("Discovered feature names from scaler.feature_names_in_")
            except Exception:
                feature_names = None

    if feature_names is None and hasattr(model, "feature_names_in_"):
        try:
            feature_names = list(getattr(model, "feature_names_in_") )
            logger.info("Discovered feature names from model.feature_names_in_")
        except Exception:
            feature_names = None

    # xgboost booster
    if feature_names is None:
        try:
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                if hasattr(booster, "feature_names") and booster.feature_names is not None:
                    feature_names = list(booster.feature_names)
                    logger.info("Discovered feature names from xgboost booster.feature_names")
        except Exception:
            pass

    return model, scaler, feature_names


def get_expected_features(model: Any, scaler: Any, saved_feature_names: Optional[List[str]]) -> List[str]:
    if saved_feature_names:
        return list(saved_feature_names)

    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(getattr(scaler, "feature_names_in_") )

    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_") )

    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            if hasattr(booster, "feature_names") and booster.feature_names is not None:
                return list(booster.feature_names)
    except Exception:
        pass

    raise ValueError(
        "Could not determine expected feature names. When training, save the feature names together with the model."
    )


def align_input(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df_aligned = df.reindex(columns=expected_features, fill_value=0)

    # try numeric conversion where possible
    for col in df_aligned.columns:
        try:
            df_aligned[col] = pd.to_numeric(df_aligned[col], errors="coerce")
        except Exception:
            pass
    # Fill numeric NaNs with 0
    df_aligned = df_aligned.fillna(0)
    return df_aligned


def predict_from_df(model: Any, scaler: Optional[Any], df: pd.DataFrame):
    X = df.values
    if scaler is not None:
        try:
            X = scaler.transform(df)
        except Exception:
            X = scaler.transform(df.values)

    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs_all = model.predict_proba(X)
            if probs_all.shape[1] == 2:
                probs = probs_all[:, 1]
            else:
                probs = probs_all.tolist()
        except Exception:
            probs = None

    results = []
    for i in range(len(df)):
        r = {"prediction": int(preds[i]) if hasattr(preds[i], "__int__") else preds[i]}
        if probs is not None:
            r["probability"] = float(probs[i]) if hasattr(probs[i], "__float__") else probs[i]
        results.append(r)
    return results


# --- Streamlit UI ---
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction — Streamlit")
st.write("Upload a model (pickle) or provide a model path, then provide input via form or CSV.")

# Sidebar: model controls
st.sidebar.header("Model & Settings")
uploaded_model_file = st.sidebar.file_uploader("Upload model (.pkl/.joblib)", type=["pkl", "joblib"], accept_multiple_files=False)
model_path_input = st.sidebar.text_input("Or enter model path (server) or leave empty to use default:", value=DEFAULT_MODEL_PATH)
load_button = st.sidebar.button("Load model")

MODEL = None
SCALER = None
FEATURES = None
load_error = None

if uploaded_model_file is not None:
    try:
        b = uploaded_model_file.read()
        MODEL, SCALER, FEATURES = load_model_from_bytes(b)
        st.sidebar.success("Model loaded from uploaded file")
    except Exception as e:
        load_error = str(e)
        st.sidebar.error(f"Failed to load uploaded model: {e}")

elif load_button:
    # try to load from path
    try:
        path = model_path_input.strip() or DEFAULT_MODEL_PATH
        MODEL, SCALER, FEATURES = load_model_from_path(path)
        st.sidebar.success(f"Model loaded from {path}")
    except Exception as e:
        load_error = str(e)
        st.sidebar.error(f"Failed to load model: {e}")

else:
    # attempt to auto-load default path if exists
    try:
        if os.path.exists(DEFAULT_MODEL_PATH):
            MODEL, SCALER, FEATURES = load_model_from_path(DEFAULT_MODEL_PATH)
            st.sidebar.info(f"Auto-loaded model from {DEFAULT_MODEL_PATH}")
    except Exception as e:
        load_error = str(e)

if MODEL is None and load_error is None:
    st.sidebar.info("No model loaded yet. Upload or load one to start predicting.")

if load_error:
    st.sidebar.write("Error details:")
    st.sidebar.code(load_error)

# Main UI
if MODEL is None:
    st.warning("Model not loaded. Upload a model file or load from a path in the sidebar.")
    st.stop()

# Determine expected features
try:
    EXPECTED_FEATURES = get_expected_features(MODEL, SCALER, FEATURES)
except Exception as e:
    st.error(f"Could not determine expected features: {e}")
    st.stop()

st.success(f"Model ready. Expected features count: {len(EXPECTED_FEATURES)}")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Feature list")
    st.write(EXPECTED_FEATURES)

with col1:
    st.subheader("Predict for a single record")
    with st.form(key="single_record_form"):
        inputs = {}
        # Render simple inputs; all as text inputs, convert to numeric where possible
        for feat in EXPECTED_FEATURES:
            # Choose widget based on name heuristics
            if any(x in feat.lower() for x in ["charge", "monthly", "total", "tenure", "count", "num", "age"]):
                val = st.number_input(label=feat, value=0.0, step=1.0, format="%.4f")
            elif any(x in feat.lower() for x in ["senior", "is_", "has_", "flag", "indicator"]):
                val = st.selectbox(label=feat, options=[0, 1], index=0)
            else:
                val = st.text_input(label=feat, value="0")
            inputs[feat] = val

        submit = st.form_submit_button("Predict")

    if submit:
        # Build a DataFrame
        df_single = pd.DataFrame([inputs])
        # Try convert numeric-like strings
        for c in df_single.columns:
            df_single[c] = pd.to_numeric(df_single[c], errors="ignore")

        df_aligned = align_input(df_single, EXPECTED_FEATURES)
        results = predict_from_df(MODEL, SCALER, df_aligned)
        st.write("**Prediction**")
        st.json(results[0])

st.markdown("---")

st.subheader("Batch predictions (CSV)")
uploaded_csv = st.file_uploader("Upload CSV with features (columns should match model features)", type=["csv"]) 
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.write("Uploaded data (first rows):")
        st.dataframe(df.head())

        df_aligned = align_input(df, EXPECTED_FEATURES)
        results = predict_from_df(MODEL, SCALER, df_aligned)

        preds = [r["prediction"] for r in results]
        probs = [r.get("probability") for r in results]
        df_out = df.copy()
        df_out["prediction"] = preds
        df_out["probability"] = probs

        st.write("Predictions:")
        st.dataframe(df_out.head())

        # download
        csv_buf = io.StringIO()
        df_out.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        st.download_button("Download predictions as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    except Exception as exc:
        st.error(f"Failed to read/process CSV: {exc}")

st.markdown("---")

st.info("Tips: \n- If your model expects one-hot encoded categorical columns, make sure your CSV/form provides those columns exactly as during training.\n- Best practice: when training, save {'model','scaler','feature_names'} together using pickle so this app can reliably align inputs.")

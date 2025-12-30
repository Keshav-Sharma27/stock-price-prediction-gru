import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    layout="centered"
)

st.title("📈 Stock Price Prediction (GRU Model)")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_gru_model():
    return load_model("models/gru_model.h5", compile=False)

model = load_gru_model()
st.success("✅ GRU model loaded successfully!")

# ------------------ CLEAN DATA ------------------
def clean_and_prepare_data(df):
    df.columns = df.columns.str.strip()

    # Parse Date if exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")

    # Force numeric conversion
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Close"])
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------ INDICATORS ------------------
def add_indicators(df):
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📤 Upload RAW stock CSV file", type=["csv"])

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.write(raw_data.tail())

    data = clean_and_prepare_data(raw_data)
    data = add_indicators(data)

    st.subheader("📊 Data After Feature Engineering")
    st.write(data.tail())

    required_cols = ["Close", "MA7", "MA21", "RSI14", "MACD"]
    features = data[required_cols]

    # ------------------ PREDICTION ------------------
    if st.button("🔮 Predict Next Day Price"):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)

        if len(scaled_data) < 60:
            st.error("❌ At least 60 rows are required for prediction")
            st.stop()

        last_60 = scaled_data[-60:]
        X_input = np.expand_dims(last_60, axis=0)

        prediction = model.predict(X_input)

        dummy = np.zeros((1, scaled_data.shape[1]))
        dummy[0, 0] = prediction[0, 0]

        predicted_price = scaler.inverse_transform(dummy)[0, 0]

        st.success(
            f"📈 Predicted NEXT DAY Closing Price: ₹ {predicted_price:.2f}"
        )

        st.info(
            "⚠ This prediction is based on historical patterns, not financial advice."
        )

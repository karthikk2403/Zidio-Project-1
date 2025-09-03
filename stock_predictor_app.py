# Pro Stock Predictor - Final Version with Flexible History
# Last Updated: 2025-08-27
# Location Context: Vijayawada, Andhra Pradesh, India

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "users_tmp.sqlite"

# --- Database & User Authentication Functions ---
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS users(
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE,
                     salt TEXT,
                     pwd_hash TEXT,
                     name TEXT,
                     email TEXT,
                     bio TEXT,
                     created_at TEXT)""")
    conn.commit()
    return conn

def hash_pwd(pwd, salt):
    return hashlib.sha256((salt + pwd).encode()).hexdigest()

def create_user(username, pwd, name="", email=""):
    conn = get_conn()
    salt = os.urandom(16).hex()
    h = hash_pwd(pwd, salt)
    try:
        conn.execute("INSERT INTO users(username,salt,pwd_hash,name,email,bio,created_at) VALUES(?,?,?,?,?,?,?)",
                     (username, salt, h, name, email, "", datetime.utcnow().isoformat()))
        conn.commit()
        return True
    except Exception: return False

def auth_user(username, pwd):
    conn = get_conn()
    cur = conn.execute("SELECT id, username, salt, pwd_hash, name, email, bio FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    if not row: return None
    if hash_pwd(pwd, row[2]) == row[3]:
        return {"id": row[0], "username": row[1], "name": row[4], "email": row[5], "bio": row[6]}
    return None

def update_profile(uid, name, email, bio):
    conn = get_conn()
    conn.execute("UPDATE users SET name=?, email=?, bio=? WHERE id=?", (name, email, bio, uid))
    conn.commit()

# --- Stock Data & Constants ---
POPULAR = [
    ("AAPL","Apple"),("MSFT","Microsoft"),("GOOGL","Alphabet"),("AMZN","Amazon"),("META","Meta"),("TSLA","Tesla"),
    ("NVDA","NVIDIA"),("BRK-B","Berkshire Hathaway"),("JPM","JPMorgan"),("V","Visa"),("JNJ","Johnson & Johnson"),
    ("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy"), ("HDFCBANK.NS", "HDFC Bank"),
    ("INFY.NS", "Infosys"), ("ICICIBANK.NS", "ICICI Bank"),
    ("ADBE","Adobe"),("NFLX","Netflix"),("DIS","Disney"),("CSCO","Cisco"),("ORCL","Oracle"),("NKE","Nike"),
]

# --- App Configuration & Styling ---
st.set_page_config(page_title="Stock Pro", layout="wide", page_icon="📈")
st.markdown("""
<style>
.main {background: radial-gradient(1200px 600px at 10% 10%, rgba(99,102,241,.06), transparent), radial-gradient(900px 400px at 90% 20%, rgba(16,185,129,.04), transparent), linear-gradient(180deg, #071029 0%, #071a2e 100%); color:#e6e8ff}
h1 {font-weight:700}
.glass{background:rgba(255,255,255,.03); backdrop-filter:blur(6px); border:1px solid rgba(255,255,255,.04); border-radius:12px; padding:12px}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "user" not in st.session_state: st.session_state.user = None
if "page" not in st.session_state: st.session_state.page = "Login/Register"

# --- Data Fetching & Machine Learning Functions ---
@st.cache_data(show_spinner=False)
def fetch_prices(ticker, years=0):
    end = datetime.now()
    start = None
    # If years are specified, calculate start date. Otherwise, start is None to get all data.
    if years > 0:
        start = end - timedelta(days=int(years * 365.25))
    
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    
    if df is None or df.empty: return pd.DataFrame()
        
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
    df = df.reset_index()
    
    df.rename(columns={'Date': 'Date', 'Close': 'Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Date', 'Close'], inplace=True)
    df.sort_values(by='Date', inplace=True)
    df.drop_duplicates(subset='Date', keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df[['Date', 'Close']]

@st.cache_resource(show_spinner=False)
def train_models(df):
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    models = {}
    series = train_df["Close"].astype(float)

    try: models["ARIMA"] = ARIMA(series, order=(1,1,1)).fit()
    except: models["ARIMA"] = None
    try: models["SARIMA"] = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,5)).fit(disp=False)
    except: models["SARIMA"] = None
        
    p_df = train_df.rename(columns={"Date": "ds", "Close": "y"})
    try:
        p = Prophet(); p.fit(p_df)
        models["Prophet"] = p
    except: models["Prophet"] = None
            
    scaler = MinMaxScaler(feature_range=(0,1))
    train_close_prices = train_df['Close'].values.reshape(-1,1)
    scaler.fit(train_close_prices)
    
    full_scaled_data = scaler.transform(df['Close'].values.reshape(-1,1))
    ts = 60
    train_scaled_data = full_scaled_data[:train_size]
    
    X_train, y_train = [], []
    for i in range(ts, len(train_scaled_data)):
        X_train.append(train_scaled_data[i-ts:i, 0])
        y_train.append(train_scaled_data[i, 0])
        
    if X_train:
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        try:
            lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)), LSTM(50), Dense(1)])
            lstm.compile(optimizer="adam", loss="mean_squared_error")
            lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            models["LSTM"] = lstm
        except: models["LSTM"] = None
    else: models["LSTM"] = None

    models["scaler"], models["ts"], models["train_size"] = scaler, ts, train_size
    return models

def lstm_insample_point(df, mdl, scaler, ts, idx):
    if mdl is None or idx < ts: return None
    try:
        window_unscaled = df[["Close"]].values[idx-ts:idx]
        window_scaled = scaler.transform(window_unscaled)
        X = np.expand_dims(window_scaled, axis=0)
        pred_scaled = mdl.predict(X, verbose=0)
        return float(scaler.inverse_transform(pred_scaled)[0][0])
    except: return None

def predict_values(df, models, target_date):
    last_date = df["Date"].max().date()
    first_date = df["Date"].min().date()
    is_inside = first_date <= target_date <= last_date
    
    idx = None
    if is_inside:
        date_match = df[df['Date'].dt.date == target_date]
        if not date_match.empty: idx = date_match.index[0]

    out = {}
    for model_name, model in models.items():
        if model is None or model_name in ["scaler", "ts", "train_size"]: continue
        try:
            pred = None
            if is_inside:
                if idx is None: pred = None
                elif model_name in ["ARIMA", "SARIMA"]: pred = model.predict(start=idx, end=idx).iloc[0]
                elif model_name == "Prophet":
                    f = pd.DataFrame({"ds": [pd.to_datetime(target_date)]}); pred = model.predict(f)["yhat"].iloc[0]
                elif model_name == "LSTM": pred = lstm_insample_point(df, model, models["scaler"], models["ts"], idx)
            else:
                days = max(1, (target_date - last_date).days)
                if model_name in ["ARIMA", "SARIMA"]: pred = model.forecast(steps=days).iloc[-1]
                elif model_name == "Prophet":
                    f = model.make_future_dataframe(periods=days, freq='D'); pred = model.predict(f)["yhat"].iloc[-1]
                elif model_name == "LSTM":
                    ts, scaler = models["ts"], models["scaler"]
                    last_window = df["Close"].values[-ts:]
                    current_input = scaler.transform(last_window.reshape(-1, 1)).reshape(1, ts, 1)
                    for _ in range(days):
                        pred_scaled = model.predict(current_input, verbose=0)
                        new_pred_reshaped = pred_scaled.reshape(1, 1, 1)
                        current_input = np.append(current_input[:, 1:, :], new_pred_reshaped, axis=1)
                    pred = scaler.inverse_transform(pred_scaled)[0][0]
            out[model_name] = float(pred) if pred is not None else None
        except: out[model_name] = None
    return out

# --- UI View Functions ---
def header():
    st.markdown("<h1 style='font-weight:700'>📈 Stock Pro </h1>", unsafe_allow_html=True)
    if "user" in st.session_state and st.session_state.user:
        u = st.session_state.user
        st.markdown(f"<div style='text-align:right;'>👤 {u.get('name') or u.get('username')}</div>", unsafe_allow_html=True)

def login_register_view():
    st.title("Welcome")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Login"); username = st.text_input("Username", key="login_user"); password = st.text_input("Password", type="password", key="login_pwd")
            if st.button("Login", key="login_btn", type="primary"):
                user = auth_user(username.strip(), password)
                if user: st.session_state.user = user; st.session_state.page = "Dashboard"; st.success("Logged in successfully!"); st.rerun()
                else: st.error("Invalid credentials")
    with c2:
        with st.container(border=True):
            st.subheader("Register"); name = st.text_input("Full name", key="reg_name"); email = st.text_input("Email", key="reg_email"); username2 = st.text_input("Username", key="reg_user"); pwd1 = st.text_input("Password", type="password", key="reg_pwd1"); pwd2 = st.text_input("Confirm Password", type="password", key="reg_pwd2")
            if st.button("Create Account", key="reg_btn"):
                if not username2 or not pwd1 or pwd1 != pwd2: st.error("Please fill all fields and ensure passwords match.")
                else:
                    ok = create_user(username2.strip(), pwd1, name.strip(), email.strip())
                    if ok: st.success("Account created. You can now login.")
                    else: st.error("Username already exists.")

def profile_view():
    st.title("User Profile")
    u = st.session_state.user
    with st.container(border=True):
        name = st.text_input("Name", value=u.get("name") or "", key="profile_name"); email = st.text_input("Email", value=u.get("email") or "", key="profile_email")
        if st.button("Save Profile", key="save_profile", type="primary"):
            update_profile(u["id"], name.strip(), email.strip(), ""); st.session_state.user.update({'name': name.strip(), 'email': email.strip()}); st.success("Profile updated")

def dashboard_view():
    st.title("Dashboard")
    with st.container(border=True):
        tickers = [f"{t} · {n}" for t, n in POPULAR]
        c1, c2, c3 = st.columns([0.45, 0.35, 0.2])
        with c1: pick = st.selectbox("Popular tickers", tickers, key="dash_pick")
        with c2: custom = st.text_input("Or enter custom ticker", key="dash_custom", help="For non-US stocks, add exchange suffix. E.g., RELIANCE.NS")
        with c3:
            # --- FEATURE UPDATE: Replaced slider with selectbox ---
            year_options = {"5 Years": 5, "10 Years": 10, "15 Years": 15, "20 Years": 20, "All Available": 0}
            selected_period = st.selectbox("Select Data History", options=list(year_options.keys()), index=1)
            years_to_fetch = year_options[selected_period]
        
        t = custom.strip().upper() if custom else pick.split(" · ")[0]
        
        if st.button("Load & Train Models", key="load_btn", type="primary"):
            with st.spinner(f"Fetching {t} data and training models..."):
                df = fetch_prices(t, years_to_fetch)
                if df.empty:
                    st.error(f"No data found for ticker '{t}'.")
                else:
                    st.session_state.ticker, st.session_state.df = t, df
                    st.session_state.models = train_models(df)
                    st.success("Data loaded and models trained!")
                    all_model_names = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
                    trained = [name for name, model in st.session_state.models.items() if model and name in all_model_names]
                    failed = [name for name in all_model_names if name not in trained]
                    if failed: st.warning(f"Note: The following model(s) could not be trained, likely due to insufficient historical data: **{', '.join(failed)}**")

    if "df" in st.session_state and st.session_state.df is not None:
        st.subheader(f"Historical Data for {st.session_state.ticker}")
        df, t = st.session_state.df, st.session_state.ticker
        last, prev = df.iloc[-1]["Close"], df.iloc[-2]["Close"] if len(df) > 1 else 0
        
        c1, c2, c3 = st.columns(3)
        is_inr = ".NS" in t or ".BO" in t
        c1.metric("Last Close Price", f"₹{last:,.2f}" if is_inr else f"${last:,.2f}", f"{last - prev:,.2f}")
        c2.metric("Day Change", f"{(last - prev) / prev * 100:+.2f}%" if prev else "0.00%")
        c3.metric("Data Points", f"{len(df):,}")
        
        fig, ax = plt.subplots(); ax.plot(df["Date"], df["Close"]); ax.set_title(f"{t} Price History"); ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

def predictor_view():
    st.title("Price Predictor")
    if "df" not in st.session_state or st.session_state.df is None:
        st.info("Please load a ticker on the Dashboard page first."); return
        
    df, models, t = st.session_state.df, st.session_state.models, st.session_state.ticker
    
    with st.container(border=True):
        all_models = [m for m in ["ARIMA", "SARIMA", "Prophet", "LSTM"] if m in models]
        c1, c2 = st.columns([0.65, 0.35])
        with c1: pick = st.multiselect("Select models for prediction", all_models, default=all_models)
        with c2: target = st.date_input("Select prediction date", value=df["Date"].max().date() + timedelta(days=7))
        
        if st.button("Predict", type="primary") and pick:
            is_weekend = target.weekday() >= 5
            is_future = target > df['Date'].max().date()
            is_in_data = not df[df['Date'].dt.date == target].empty
            
            if is_weekend:
                st.warning(f"**Non-Trading Day:** The selected date ({target.strftime('%d-%b-%Y')}) is a weekend.")
                if 'predictions' in st.session_state: del st.session_state['predictions']
            elif not is_in_data and not is_future:
                st.warning(f"**Non-Trading Day:** The selected date ({target.strftime('%d-%b-%Y')}) was likely a market holiday.")
                if 'predictions' in st.session_state: del st.session_state['predictions']
            else:
                with st.spinner("Generating predictions..."):
                    st.session_state.predictions = predict_values(df, {k: models[k] for k in pick}, target)

    if "predictions" in st.session_state and st.session_state.predictions:
        st.subheader(f"Predictions for {target.strftime('%d-%b-%Y')}")
        preds = st.session_state.predictions
        cols = st.columns(len(preds) or 1)
        is_inr = ".NS" in t or ".BO" in t
        for i, (model, value) in enumerate(preds.items()):
            cols[i].metric(model, f"₹{value:,.2f}" if is_inr and value else f"${value:,.2f}" if value else "N/A")

        fig, ax = plt.subplots(); ax.plot(df["Date"], df["Close"], label="History"); ax.axvline(pd.to_datetime(target), ls="--", color='red')
        for k, v in preds.items():
            if v: ax.scatter(pd.to_datetime(target), v, label=f"{k}", s=90, zorder=5)
        ax.legend(); ax.grid(True, alpha=0.3); st.pyplot(fig, use_container_width=True)

# --- Main App Logic & Navigation ---
header()

with st.sidebar:
    st.markdown("### Navigation")
    if st.session_state.user:
        pages = ["Dashboard", "Predictor", "Profile"]
        page_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0
        page = st.radio("Go to", pages, index=page_index, key="nav_radio")
        if page != st.session_state.page: st.session_state.page = page; st.rerun()
        if st.button("Logout", key="logout_btn"): st.session_state.page = "Logout"; st.rerun()
    else: st.session_state.page = "Login/Register"

if st.session_state.page == "Login/Register": login_register_view()
elif st.session_state.page == "Dashboard": dashboard_view()
elif st.session_state.page == "Predictor": predictor_view()
elif st.session_state.page == "Profile": profile_view()
elif st.session_state.page == "Logout":
    for key in list(st.session_state.keys()):
        if key not in ['page']: del st.session_state[key]
    st.session_state.user = None
    st.rerun()
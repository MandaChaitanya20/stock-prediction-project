import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Page config
st.set_page_config(page_title="Netflix Stock Dashboard", layout="wide")

# 🎨 Colorful UI
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1 {
    text-align: center;
    color: #00f5d4;
}
.stMetric {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Netflix Stock Price Prediction Dashboard")

# ✅ LOAD DATA (ROBUST)
@st.cache_data
def load_data():
    data = pd.read_csv("nflx.csv")

    # Clean column names
    data.columns = [col.strip().lower() for col in data.columns]

    # Detect date column
    if 'date' in data.columns:
        date_col = 'date'
    else:
        st.error(f"❌ Date column not found. Columns: {list(data.columns)}")
        st.stop()

    # Convert date
    data[date_col] = pd.to_datetime(data[date_col])

    # Set index
    data.set_index(date_col, inplace=True)

    return data

data = load_data()

# Sidebar
st.sidebar.header("⚙️ Settings")
days = st.sidebar.slider("Predict next days", 1, 30, 7)

# 📊 Metrics
latest_price = float(data["close"].iloc[-1])
previous_price = float(data["close"].iloc[-2])
change = latest_price - previous_price

col1, col2 = st.columns(2)
col1.metric("💰 Latest Price", f"${latest_price:.2f}")
col2.metric("📉 Change", f"{change:.2f}")

# 📈 LINE GRAPH
st.subheader("📊 Stock Price Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['close'],
    mode='lines',
    name='Actual Price',
    line=dict(color='#00f5d4', width=3)
))

fig.update_layout(
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font=dict(color="white"),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# 🤖 PREDICTION (Linear Regression)

st.subheader("🤖 Price Prediction")

data_reset = data.reset_index()
data_reset['Days'] = np.arange(len(data_reset))

X = data_reset[['Days']]
y = data_reset['close']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(data_reset), len(data_reset) + days).reshape(-1, 1)
predictions = model.predict(future_days)

# Future dates
last_date = data_reset.iloc[-1, 0]
future_dates = pd.date_range(last_date, periods=days+1)[1:]

# 📈 Prediction Graph
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data.index,
    y=data['close'],
    mode='lines',
    name='Actual',
    line=dict(color='#00f5d4', width=3)
))

fig2.add_trace(go.Scatter(
    x=future_dates,
    y=predictions,
    mode='lines',
    name='Predicted',
    line=dict(color='#ff006e', width=3, dash='dash')
))

fig2.update_layout(
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font=dict(color="white"),
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# Show data
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# Footer
st.markdown("---")
st.markdown("<h4 style='text-align:center; color:gray;'>🚀 Built by Chaithuu</h4>", unsafe_allow_html=True)
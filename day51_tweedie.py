import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="Day 51 - Tweedie Regression", layout="wide")
st.title("📊 Day 51 — Tweedie Regression: Insurance Risk Modeling")

st.markdown("""
The **Tweedie Regressor** is a type of Generalized Linear Model (GLM) useful in **insurance and risk modeling**.  
It handles **zero-inflated continuous outcomes** like claim amounts (many zeros, some large values).
""")

# Generate synthetic insurance-like dataset
@st.cache_data
def make_data(n_samples=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 70, size=n_samples)
    car_age = rng.randint(0, 15, size=n_samples)
    exposure = rng.uniform(0.5, 1.0, size=n_samples)  # policy exposure (risk duration)
    # Claim severity with many zeros
    freq = rng.poisson(lam=0.2, size=n_samples)  # mostly zero claims
    severity = rng.gamma(shape=2.0, scale=500, size=n_samples)
    claim_amount = freq * severity
    X = pd.DataFrame({"age": age, "car_age": car_age, "exposure": exposure})
    return X, claim_amount

X, y = make_data()

st.subheader("📂 Dataset Preview")
st.write(pd.DataFrame(X).assign(claim_amount=y).head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tweedie Regressor Parameters
st.sidebar.header("⚙️ Tweedie Parameters")
power = st.sidebar.selectbox("Tweedie Power (distribution)", [0, 1, 1.5, 2], index=2,
                             help="0=Normal, 1=Poisson, (1<p<2)=Compound Poisson-Gamma, 2=Gamma")
alpha = st.sidebar.slider("Alpha (regularization)", 0.0, 1.0, 0.1, 0.01)

# Fit model
model = TweedieRegressor(power=power, alpha=alpha, link="log")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"- MSE: {mse:.2f}")
st.write(f"- MAE: {mae:.2f}")
st.write(f"- R² Score: {r2:.3f}")

# Plot Actual vs Predicted
st.subheader("🔮 Actual vs Predicted Claims")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([0, max(y_test)], [0, max(y_test)], "r--")
ax.set_xlabel("Actual Claim Amount")
ax.set_ylabel("Predicted Claim Amount")
st.pyplot(fig)

# Download Results
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
csv = results.to_csv(index=False)
st.download_button("Download Predictions CSV", csv, "tweedie_predictions.csv", "text/csv")

st.success("✅ Tweedie Regression model trained and evaluated successfully!")

# =========================================================
# 🎮 GameVision - Prophet Forecasting Model Training Script
# =========================================================

import pandas as pd
from prophet import Prophet
import joblib
import matplotlib.pyplot as plt

print("🚀 Starting Prophet model training...")

# 1️⃣ Load Prophet-ready dataset
df = pd.read_csv('../Data/prophet_ready.csv')
print("✅ Dataset loaded successfully.")
print(df.head())

# 2️⃣ Initialize Prophet model
model = Prophet(
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
)

# 3️⃣ Train the model
model.fit(df)
print("✅ Model trained successfully.")

# 4️⃣ Create future dataframe (next 5 years)
future = model.make_future_dataframe(periods=5, freq='Y')
forecast = model.predict(future)

# 5️⃣ Save the model
joblib.dump(model, '../Models/prophet_forecast.pkl')
print("💾 Model saved to ../Models/prophet_forecast.pkl")

# 6️⃣ Plot forecast
fig1 = model.plot(forecast)
plt.title('📈 Global Video Game Sales Forecast')
plt.xlabel('Year')
plt.ylabel('Sales (Millions)')
plt.show()

# 7️⃣ Plot forecast components (trends, seasonality)
fig2 = model.plot_components(forecast)
plt.show()

print("🎯 Forecasting complete! Model and visualizations ready.")

# 🎮 GameVision – Smart Video Game Sales Intelligence Dashboard

**GameVision** is a data-driven analytics platform that predicts video game sales and visualizes global market trends.  
Built using **Python**, **Prophet**, and **Streamlit**, it combines forecasting, interactive dashboards, and explainable AI to help studios and publishers make smarter decisions about genres, platforms, and regions.

---

## 🚀 Features

- 📊 **Interactive Streamlit Dashboard** – Filter by genre and platform  
- 🔮 **Prophet-Based Forecasting** – Predict global sales for future years  
- 🌍 **Market Insights** – Analyze global and regional performance  
- 🤖 **Explainable AI Ready** – Integrate SHAP for model interpretation  
- 💾 **Kaggle Dataset Integration** – Transparent and reproducible results  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python 3.12 |
| Dashboard | Streamlit, Plotly |
| Forecasting | Prophet |
| Data Processing | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| ML Explainability | SHAP (optional) |

---

## 🗂️ Project Structure
GameVision/
├── App/
│ ├── app.py # Main Streamlit dashboard
│ ├── plots.py # Visualization helpers
│ ├── utils.py # Data loading & cleaning
│
├── Data/
│ ├── vgsales.csv # Raw Kaggle dataset
│ ├── vgsales_clean.csv # Cleaned dataset
│ └── prophet_ready.csv # Prophet-ready dataset
│
├── Models/
│ ├── train_prophet.py # Model training script
│ └── prophet_forecast.pkl # Saved Prophet model
│
├── Notebooks/
│ └── data_analysis.py # EDA and preprocessing
│
├── requirements.txt
└── README.md




---

## ⚙️ Installation & Setup

### 1️Clone the Repository
```bash
git clone https://github.com/<your-username>/GameVision.git
cd GameVision

## Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Mac/Linux


Install dependencies
pip install -r requirements.txt


4️⃣ Run the Dashboard
cd App
streamlit run app.py

🧩 Future Enhancements

Regional trend comparison tabs

SHAP explainable AI integration

Cloud deployment via Streamlit Cloud

Interactive forecast range slider


🏆 Author

Bhanoday Kurma
📧 bhanodaykurma27@gmail.com

📍 Bangalore, India

📜 License

This project is open-source under the MIT License
.




# ======================================================
# 🎮 GameVision - Data Analysis & Preprocessing Notebook
# ======================================================

# 🧩 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Setup styles
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

print("✅ Libraries imported successfully.")


# 📂 2. Load Dataset
df = pd.read_csv('../data/vgsales.csv')
print("✅ Dataset loaded successfully.")
df.head()


# 🔍 3. Basic Info & Overview
print("📘 Dataset Info:")
print(df.info())

print("\n📊 Summary Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())


# 🧹 4. Data Cleaning
# Drop missing values for critical columns
print(df.columns.tolist())
df.dropna(subset=['Year_of_Release', 'Genre', 'Platform', 'Global_Sales'], inplace=True)

# Convert Year column to integer
df['Year_of_Release'] = df['Year_of_Release'].astype(int)

# Filter to focus on modern data (optional)
df = df[df['Year_of_Release'] >= 2000]

# Reset index
df.reset_index(drop=True, inplace=True)
print("✅ Data cleaned successfully!")


# 🧾 5. Feature Summary
print("🎮 Unique Genres:", df['Genre'].nunique())
print("🕹️ Unique Platforms:", df['Platform'].nunique())
print("📅 Year Range:", df['Year_of_Release'].min(), "-", df['Year_of_Release'].max())


# 📈 6. Global Sales Trend Over Time
yearly_sales = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
yearly_sales.rename(columns={'Year_of_Release': 'Year'}, inplace=True)

plt.figure(figsize=(10,5))
sns.lineplot(data=yearly_sales, x='Year', y='Global_Sales', marker='o', color='dodgerblue')
plt.title('🎮 Global Video Game Sales Over Time')
plt.xlabel('Year')
plt.ylabel('Total Global Sales (Millions)')
plt.show()


# 🕹️ 7. Top 10 Game Genres by Global Sales
top_genres = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_genres.index, y=top_genres.values, palette='coolwarm')
plt.title('Top 10 Game Genres by Global Sales')
plt.ylabel('Total Global Sales (Millions)')
plt.xticks(rotation=45)
plt.show()


# 💻 8. Top 10 Platforms by Global Sales
top_platforms = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_platforms.index, y=top_platforms.values, palette='viridis')
plt.title('Top 10 Platforms by Global Sales')
plt.ylabel('Total Global Sales (Millions)')
plt.xticks(rotation=45)
plt.show()


# 🌍 9. Regional Insights (Optional)
region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
if all(col in df.columns for col in region_cols):
    region_sales = df[region_cols].sum().sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    sns.barplot(x=region_sales.index, y=region_sales.values, palette='mako')
    plt.title('Regional Sales Distribution')
    plt.ylabel('Total Sales (Millions)')
    plt.show()


# 🔢 10. Correlation Heatmap (Optional)
if all(col in df.columns for col in region_cols):
    corr = df[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].corr()
    plt.figure(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Correlation between Regional Sales')
    plt.show()


# 📊 11. Interactive Visualization with Plotly
fig = px.line(yearly_sales, x='Year', y='Global_Sales',
              title='Interactive Global Sales Over Time',
              markers=True)
fig.update_traces(line=dict(width=4))
fig.show()


# 💾 12. Export Cleaned Data
df.to_csv('../data/vgsales_clean.csv', index=False)
print("✅ Cleaned dataset saved to ../data/vgsales_clean.csv")


# 🔮 13. Prophet-Ready Dataset
prophet_df = yearly_sales.rename(columns={'Year': 'ds', 'Global_Sales': 'y'})
prophet_df.to_csv('../data/prophet_ready.csv', index=False)
print("✅ Prophet dataset ready: ../data/prophet_ready.csv")


# ✅ 14. Summary
print("\n🧭 Data Analysis Completed!")
print("Cleaned rows:", len(df))
print("Years covered:", df['Year_of_Release'].min(), "-", df['Year_of_Release'].max())
print("Files created:")
print("- ../data/vgsales_clean.csv")
print("- ../data/prophet_ready.csv")

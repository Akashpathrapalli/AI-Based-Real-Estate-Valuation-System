#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
#.\venv\Scripts\Activate.ps1
# py -3.13 -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit xgboost joblib
# pip install -r requirements.txt

# Import necessary libraries
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('housing.csv')

# Look at first 5 rows
print(df.head())

# Get summary info
print(df.info())

# Get statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

# Fill missing values with mean of column
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Fill categorical columns with mode (most common value)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("✅ Missing values handled successfully!")
print(df.info())

# If dataset has categorical columns (like location), convert them into numbers
df = pd.get_dummies(df, drop_first=True)

# Split Features (X) and Target (y)
X = df.drop('SalePrice', axis=1)   # Features
y = df['SalePrice']                # Target

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Correlation Heatmap
corr_matrix = df.corr(numeric_only=True)
top_corr_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(11).index
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_corr_features].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True,
    linewidths=0.5, annot_kws={"size": 10})
plt.title("Top 10 Features Correlated with SalePrice", fontsize=16, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

# 2. Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['SalePrice'], kde=True, bins=40, color="skyblue")
plt.title("Distribution of House Prices", fontsize=16)
plt.xlabel("Sale Price")
plt.ylabel("Count")
plt.show()

# 3. Histogram of SalePrice
plt.figure(figsize=(8,6))
plt.hist(df['SalePrice'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plots with key features
if 'GrLivArea' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], alpha=0.6)
    plt.title("Price vs. Living Area (GrLivArea)", fontsize=16)
    plt.xlabel("Living Area (sqft)")
    plt.ylabel("Sale Price")
    plt.show()

if 'YearBuilt' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df['YearBuilt'], y=df['SalePrice'], alpha=0.6)
    plt.title("Price vs. Year Built", fontsize=16)
    plt.xlabel("Year Built")
    plt.ylabel("Sale Price")
    plt.show()

# 5. Bar Chart: Average SalePrice by OverallQual
avg_price_by_quality = df.groupby('OverallQual')['SalePrice'].mean()
plt.figure(figsize=(8,6))
avg_price_by_quality.plot(kind='bar', color='orange')
plt.title("Average Sale Price by Overall Quality")
plt.xlabel("Overall Quality (1-10)")
plt.ylabel("Average Sale Price")
plt.show()

# 6. Line Plot: Average Sale Price by Year Sold
avg_price_by_year = df.groupby('YrSold')['SalePrice'].mean()
plt.figure(figsize=(6,4))
avg_price_by_year.plot(kind='line', marker='o', color='red')
plt.title("Average Sale Price by Year Sold")
plt.xlabel("Year Sold")
plt.ylabel("Average Sale Price")
plt.show()

# 7. Boxplot: SalePrice by OverallCond
plt.figure(figsize=(8,6))
df.boxplot(column='SalePrice', by='OverallCond', grid=False)
plt.title("Sale Price by Overall Condition")
plt.suptitle("")
plt.xlabel("Overall Condition (1-10)")
plt.ylabel("Sale Price")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle
import os
import sys
# Save feature names for app.py
with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
    
# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Dictionary to store results
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear Regression'] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    "MAE": mean_absolute_error(y_test, y_pred_lr),
    "R2": r2_score(y_test, y_pred_lr)
}

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    "MAE": mean_absolute_error(y_test, y_pred_dt),
    "R2": r2_score(y_test, y_pred_dt)
}

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    "MAE": mean_absolute_error(y_test, y_pred_rf),
    "R2": r2_score(y_test, y_pred_rf)
}

# XGBoost
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xg_reg.fit(X_train, y_train)
y_pred_xgb = xg_reg.predict(X_test)
results['XGBoost'] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    "MAE": mean_absolute_error(y_test, y_pred_xgb),
    "R2": r2_score(y_test, y_pred_xgb)
}

# -----------------------------
# Hyperparameter tuning (RandomizedSearch)
# -----------------------------
# Random Forest tuning
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_random = RandomizedSearchCV(rf, rf_params, n_iter=5, scoring='neg_mean_squared_error',
                               cv=3, verbose=1, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

# XGBoost tuning
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb_random = RandomizedSearchCV(xg_reg, xgb_params, n_iter=5, scoring='neg_mean_squared_error',
                                cv=3, verbose=1, random_state=42, n_jobs=-1)
xgb_random.fit(X_train, y_train)
best_xgb = xgb_random.best_estimator_

# Evaluate tuned models
y_pred_best_rf = best_rf.predict(X_test)
y_pred_best_xgb = best_xgb.predict(X_test)

rf_results = {"RMSE": np.sqrt(mean_squared_error(y_test, y_pred_best_rf))}
xgb_results = {"RMSE": np.sqrt(mean_squared_error(y_test, y_pred_best_xgb))}

# Pick best model
if rf_results["RMSE"] < xgb_results["RMSE"]:
    best_model = best_rf
    model_name = "Random Forest"
else:
    best_model = best_xgb
    model_name = "XGBoost"

# Save the best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"✅ Best model ({model_name}) saved as best_model.pkl")
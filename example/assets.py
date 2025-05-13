import pandas as pd
import numpy as np
from dagster import asset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

@asset
def load_data():
    # Ruta relativa al proyecto
    file_path = "data/ToyotaCorolla.csv"
    df = pd.read_csv(file_path, encoding='latin1')
    return df

@asset
def eda(load_data: pd.DataFrame):
    print(load_data.describe())
    print("\nColumnas:", load_data.columns.tolist())
    print("\nCorrelación con 'Price':")
    print(load_data.corr(numeric_only=True)['Price'].sort_values(ascending=False))

@asset
def prepare_data(load_data: pd.DataFrame):
    features = ['Age_08_04', 'KM', 'HP', 'Doors', 'Gears', 'Weight']
    X = load_data[features]
    y = load_data['Price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

@asset
def train_model(prepare_data):
    X_train, X_test, y_train, y_test = prepare_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

@asset
def evaluate_model(train_model):
    model, X_test, y_test = train_model
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    rss = np.sum((y_test - y_pred) ** 2)

    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RSS: {rss:.2f}")

@asset
def analyze_residuals(train_model):
    model, X_test, y_test = train_model
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Análisis de los residuales")
    plt.grid(True)
    plt.show()
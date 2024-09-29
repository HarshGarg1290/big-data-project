import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

# Load the pre-trained model if available
model_path = 'trained_sales_model.pkl'
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = joblib.load(model_path)
else:
    print("Pre-trained model not found. Please train the model first.")
    model = None

# Load your dataset
df = pd.read_csv('Amazon_Sale_Report.csv', dtype={'Qty': 'int'}, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')  # Adjust format as necessary

df = df.dropna(subset=['Date', 'Amount'])  # Drop rows with NaT in Date or NaN in Amount
df.set_index('Date', inplace=True)  # Set Date as index

# Prepare data for modeling if model is not loaded
if model is None:
    # Create features (e.g., Year and Month)
    df['Year'] = df.index.year
    df['Month'] = df.index.month

    # Target variable is the sales amount
    X = df[['Year', 'Month']]
    y = df['Amount']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a regression model (using Linear Regression here)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model for future use
    joblib.dump(model, model_path)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")

# 1. Time Series Analysis: Sales Over Time
def plot_sales_over_time():
    plt.figure(figsize=(14, 7))
    df_grouped = df.groupby('Date').agg({'Amount': 'sum'}).reset_index()
    sns.lineplot(x='Date', y='Amount', data=df_grouped, color='blue')
    plt.title('Total Sales Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Sales', fontsize=14)
    plt.grid(True)
    plt.show()

# 2. Sales by Category
def plot_sales_by_category():
    plt.figure(figsize=(12, 6))
    df_category = df.groupby('Category').agg({'Amount': 'sum'}).sort_values('Amount', ascending=False).reset_index()
    sns.barplot(x='Amount', y='Category', data=df_category, palette='viridis', hue='Category', dodge=False)
    plt.title('Total Sales by Category', fontsize=16)
    plt.xlabel('Total Sales', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    plt.grid(True)
    plt.show()

# 3. Sales by Fulfillment Channel
def plot_sales_by_fulfilment():
    plt.figure(figsize=(8, 8))
    fulfilment_sales = df.groupby('Fulfilment').agg({'Amount': 'sum'}).reset_index()
    plt.pie(fulfilment_sales['Amount'], labels=fulfilment_sales['Fulfilment'], autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    plt.title('Sales Distribution by Fulfilment Channel', fontsize=16)
    plt.show()

# 4. Sales by Region (Ship-State)
def plot_sales_by_region():
    plt.figure(figsize=(14, 6))
    region_sales = df.groupby('ship-state').agg({'Amount': 'sum'}).sort_values('Amount', ascending=False).reset_index()
    sns.barplot(x='ship-state', y='Amount', data=region_sales.head(10), palette='coolwarm', hue='ship-state', dodge=False)
    plt.title('Top 10 States by Sales', fontsize=16)
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Total Sales', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# 5. Sales vs Quantity Sold
def plot_sales_vs_quantity():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Qty', y='Amount', data=df, alpha=0.6, hue='Category', palette='tab10')
    plt.title('Sales vs Quantity Sold', fontsize=16)
    plt.xlabel('Quantity Sold', fontsize=14)
    plt.ylabel('Sales Amount', fontsize=14)
    plt.grid(True)
    plt.show()

def forecast_future_sales_enhanced():
    # Generate historical sales data
    historical_sales = df.groupby('Date')['Amount'].sum().reset_index()

    # Create lagged features
    historical_sales['Lag1'] = historical_sales['Amount'].shift(1)
    historical_sales['Lag2'] = historical_sales['Amount'].shift(2)
    historical_sales.dropna(inplace=True)  # Drop rows with NaN values after lagging

    # Create features and target variable
    X = historical_sales[['Lag1', 'Lag2']]
    y = historical_sales['Amount']

    # Train a more complex model (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Prepare future dates and features
    last_sales = historical_sales.tail(2)[['Amount']].values.flatten()  # Get last two sales
    future_dates = pd.date_range(historical_sales['Date'].max() + timedelta(days=1), periods=30, freq='D')

    predictions = []

    for _ in range(30):
        # Create a DataFrame for the next prediction
        X_future = pd.DataFrame([[last_sales[0], last_sales[1]]], columns=['Lag1', 'Lag2'])  # Use DataFrame with feature names
        next_sales = model.predict(X_future)
        predictions.append(next_sales[0])

        # Update last_sales for the next prediction
        last_sales = np.array([next_sales[0], last_sales[0]])

    # Create a DataFrame for the predicted sales
    predicted_sales_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Sales': predictions
    })

    # Combine historical and predicted data for plotting
    combined_data = pd.concat([historical_sales, predicted_sales_df], ignore_index=True)

    # Plot historical vs predicted sales
    plt.figure(figsize=(12, 6))
    plt.plot(historical_sales['Date'], historical_sales['Amount'], label='Historical Sales', color='blue')
    plt.plot(predicted_sales_df['Date'], predicted_sales_df['Predicted Sales'], label='Predicted Sales', color='orange')

    plt.title('Historical vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.grid()
    plt.show()




# Call each function to generate the visualizations and forecasting
plot_sales_over_time()           # Time Series Analysis
plot_sales_by_category()         # Sales by Category
plot_sales_by_fulfilment()       # Sales by Fulfilment
plot_sales_by_region()           # Sales by Region
plot_sales_vs_quantity()         # Sales vs Quantity Sold
forecast_future_sales_enhanced()
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('Amazon_Sale_Report.csv', low_memory=False)  # Set low_memory to False

# Display the first few rows and column names to inspect date format
print(df.head())
print(df.columns)

# Ensure 'Date' is datetime with explicit format if known
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')  # Adjusting for MM-DD-YY format

# Check for NaN values in 'Date' and 'Amount'
print(f"NaN values in 'Date': {df['Date'].isna().sum()}")
print(f"NaN values in 'Amount': {df['Amount'].isna().sum()}")

# Drop rows where 'Date' or 'Amount' is NaT or NaN
df = df.dropna(subset=['Date', 'Amount'])

# Check data shape after dropping NaNs
print(f"Data shape after dropping NaNs: {df.shape}")

# Proceed only if we have data left
if df.shape[0] > 0:
    df.set_index('Date', inplace=True)

    # Create features (e.g., Year and Month) for modeling
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

    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")

    # Save the model for future use
    model_path = 'trained_sales_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Optionally, you can print model coefficients for analytics
    print("Model Coefficients:", model.coef_)
else:
    print("No data available for training the model.")

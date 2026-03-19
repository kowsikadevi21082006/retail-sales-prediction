import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Step 1: Load dataset
data = pd.read_csv("data.csv")

# Step 2: Show first few rows (for understanding)
print(data.head())

# Step 3: Basic cleaning (drop missing values)
data = data.dropna()

# Step 4: Feature Engineering
# Example: if date exists
if "date" in data.columns:
    data["date"] = pd.to_datetime(data["date"])
    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month

# Step 5: Select features (example)
features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target column from features
target = features[-1]   # last column as target
features.remove(target)

X = data[features]
y = data[target]

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 7: Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 8: Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
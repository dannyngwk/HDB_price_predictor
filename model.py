import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load a simple, clean real-world dataset (USA Housing data)
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
data = pd.read_csv(url)

# 2. Keep it incredibly simple: Use only 3 intuitive features to predict house value
# features: median_income, total_rooms, housing_median_age
features = ["median_income", "total_rooms", "housing_median_age"]
X = data[features]
y = data["median_house_value"]

# Fill any missing values quickly
X = X.fillna(X.mean())

# 3. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Save the trained model to a file so our app can use it
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 Model trained and saved successfully as 'house_model.pkl'!")
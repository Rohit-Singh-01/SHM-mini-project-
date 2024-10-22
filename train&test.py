import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
data = pd.read_csv('/home/rohit/Music/mini project/synthetic_shm_data.csv')

# Feature engineering (modify according to your data)
data['feature1'] = data['sensor1'].rolling(window=10).mean()
data['feature2'] = data['sensor2'].diff()

# Data preprocessing
data = data.dropna()
X = data[['sensor1', 'sensor2', 'sensor3', 'temp', 'humidity', 'feature1', 'feature2']]
y = data['health_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import joblib

# Load data
data = pd.read_csv('/home/rohit/Documents/synthetic_shm_data.csv')




# Data preprocessing: Selecting relevant features and handling any potential missing values
data = data.dropna()

# Define features and target
# Using the columns 'vibration', 'strain', 'displacement', 'temperature' as features
# The 'condition' is a categorical target that needs to be converted to numerical format if used for regression/classification
X = data[['vibration', 'strain', 'displacement', 'temperature']]
# Assuming 'condition' as target and we convert it to a numerical form (if it's healthy/unhealthy, use binary encoding)
data['condition'] = data['condition'].apply(lambda x: 1 if x == 'healthy' else 0)
y = data['condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Final evaluation on test set
y_pred_test = best_model.predict(X_test)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred_test, squared=False)}")

is 
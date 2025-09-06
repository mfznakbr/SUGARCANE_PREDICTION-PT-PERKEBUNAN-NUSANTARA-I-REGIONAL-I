# Import necessary libraries
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor
# Import libraries for modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Import libraries for experiment tracking
import mlflow
import mlflow.sklearn

# ==== 1. Load Data ====
train_df = pd.read_excel(r"D:\PROJECT PORTOFOLIO\SUGARCANE_PREDICTION\FILE\data_train.xlsx")
test_df = pd.read_excel(r"D:\PROJECT PORTOFOLIO\SUGARCANE_PREDICTION\FILE\testing.xlsx")

target = 'Ton' # Target variable
pipeline_path = r"D:\PROJECT PORTOFOLIO\SUGARCANE_PREDICTION\FILE\preproses_data.joblib"
file_path = r"D:\PROJECT PORTOFOLIO\SUGARCANE_PREDICTION\FILE\columns.csv"

# 1. use the preprocess_data function from preprocessing.py
preprocessor = DataPreprocessor(
    target, 
    pipeline_path, 
    file_path ,
    train_df, 
    test_df
)
X_train, X_test, y_train, y_test = preprocessor.preprocess()

print(X_train.shape)
print(X_test.shape)

# contoh 10 fitur awal
print(X_train)

# 3. Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Sugarcane Yield Prediction (GridSearch RF)")

with mlflow.start_run():
    mlflow.autolog()

    # 4. Hyperparameter Grid
    param_grid = {
        'n_estimators': [45, 50, 100],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.7]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    # 5. Train
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 6. Predict & Evaluate
    y_pred = best_model.predict(X_test) # Predictions accuracy
    mse = mean_squared_error(y_test, y_pred) # Mean Squared Error
    rmse = np.sqrt(mse) # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred) # Mean Absolute Error
    r2 = r2_score(y_test, y_pred) # Coefficient of determination R^2 score from predictions

    print("Best Hyperparameters:", best_params)
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R^2: {r2}")

    # Log metrics to MLflow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # Log the best model to MLflow
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # log the model with input example
    input_example = X_train[0].reshape(1, -1)  
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="random_forest_final_model",
        input_example=input_example,
        registered_model_name="RandomForestSugarcaneModel"
    )
"""
This script trains and evaluates an XGBoost classification model for two different lakes: Huron and Simcoe.

### Steps:
1. Define the function `train_xgb_for_lakes()`:
   - Load selected predictors for each lake
   - Load training, validation, and test datasets.
   - Handle missing values by filling them with the mean.
   - Convert datasets into DMatrix format for XGBoost.
   - Train an XGBoost model with best hyperparameters.
   - Evaluate model performance using Mean Squared Error (MSE), Accuracy, and F1-score.
   - Visualize feature importance.
2. Run the model training and evaluation for both lakes.

### Output:
- Print MSE, Accuracy, and F1-score for validation and test sets.
- Display a feature importance bar plot for the top 10 most influential features in the model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

def train_xgb_for_lakes(lake_name):
    
    predictors = {
        "HURON": [
            'Area..Filled.', 'Diameter..FD.', 'Length', 'Width', 'LON0', 'Transparency',
            'Volume..ESD.', 'MinDepth', 'LAT0', 'Aspect.Ratio', 'CiscoDen', 'Circularity',
            'WaterT', 'Intensity', 'Symmetry', 'Roughness', 'gdd2', 'CLOUD_PC', 'Geodesic.Length',
            'Compactness', 'Elongation', 'Perimeter', 'Volume..ABD.', 'Edge.Gradient', 
            'Convex.Perimeter', 'Convexity', 'Fiber.Straightness', 'Fiber.Curl', 'PRECIP', 
            'distshore', 'XANGLE'
        ],
        "SIMC": [
            'Area..ABD.', 'LON0', 'Length', 'Width', 'MaxDepth', 'Transparency', 'Symmetry', 
            'WaterT', 'Aspect.Ratio', 'Diameter..ABD.', 'Compactness', 'Elongation', 'Roughness', 
            'Convex.Perimeter', 'Intensity', 'Fiber.Straightness', 'Circularity', 'Volume..ESD.', 
            'Volume..ABD.', 'gdd2', 'Perimeter', 'Geodesic.Length', 'Edge.Gradient', 'WhitefishDen', 
            'LAT0', 'XANGLE', 'PRECIP', 'distshore', 'Exposure', 'Convexity', 'Fiber.Curl'
        ]
    }

    # Load datasets
    input_train = pd.read_csv(f"{lake_name}_input_train.csv", index_col=0)
    input_validate = pd.read_csv(f"{lake_name}_input_validate.csv", index_col=0)
    input_test = pd.read_csv(f"{lake_name}_input_test.csv", index_col=0)  
    
    output_train = pd.read_csv(f"{lake_name}_output_train.csv")[["Class"]]
    output_validate = pd.read_csv(f"{lake_name}_output_validate.csv")[["Class"]]
    output_test = pd.read_csv(f"{lake_name}_output_test.csv")[["Class"]] 

    selected_predictors = predictors[lake_name]
    input_train = input_train[selected_predictors]
    input_validate = input_validate[selected_predictors]
    input_test = input_test[selected_predictors]

    # Handle missing values
    input_train.fillna(input_train.mean(), inplace=True)
    input_validate.fillna(input_validate.mean(), inplace=True)
    input_test.fillna(input_test.mean(), inplace=True)

    # Convert to DMatrix
    dtrain = xgb.DMatrix(input_train, label=output_train.values.ravel())
    dvalidate = xgb.DMatrix(input_validate, label=output_validate.values.ravel())
    dtest = xgb.DMatrix(input_test, label=output_test.values.ravel())

    # Best hyperparameters
    best_params = {
        "HURON": {"eta": 0.03, "max_depth": 9, "subsample": 0.8, "colsample_bytree": 0.9},
        "SIMC": {"eta": 0.05, "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.8}
    }

    params = {
        "objective": "binary:logistic",  
        "eval_metric": "logloss",  
        **best_params[lake_name]
    }

    evals = [(dtrain, "train"), (dvalidate, "validate")]
    model = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=50, verbose_eval=False)

    # Predictions
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)  

    # Metrics
    accuracy = accuracy_score(output_test, y_pred_binary)
    f1 = f1_score(output_test, y_pred_binary)
    mse = mean_squared_error(output_test, y_pred)

    print(f"{lake_name} Test Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    # Feature Importance Visualization
    feature_importance = model.get_score(importance_type="weight")  
    feature_importance_df = pd.DataFrame({'Feature': feature_importance.keys(), 'Importance': feature_importance.values()})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance_df['Importance'][:10], 
                y=feature_importance_df['Feature'][:10], 
                hue=feature_importance_df['Feature'][:10],  
                palette='viridis', 
                dodge=False, 
                legend=False)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f'Top 10 Important Features in XGBoost ({lake_name})')
    plt.gca().invert_yaxis()
    plt.show()
    
    return model

if __name__ == "__main__":
    
#     grid_search
#     param_grid = {
#             "eta": [0.03, 0.05, 0.07, 0.09, 0.1],
#             "max_depth": [5, 6, 7, 8, 9, 10], 
#             "subsample": [0.7, 0.8, 0.9, 1.0], 
#             "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
#         }  
    
    print("\nTraining for HURON...")
    model_huron = train_xgb_for_lakes("HURON")
    
    print("\nTraining for SIMC...")
    model_simc = train_xgb_for_lakes("SIMC")

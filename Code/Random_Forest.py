"""
This script trains and evaluates the Random Forest classification model for two different lakes: Huron and Simcoe.

### Steps:
1. Define the function `train_and_evaluate()`:
   - Load selected predictors for each lake.
   - Load training, validation, and test datasets.
   - Handle missing values by filling them with the mean.
   - Train a Random Forest model with given hyperparameters.
   - Evaluate model performance using Mean Squared Error (MSE), Accuracy, F1-score, and AUROC.
   - Visualize feature importance.
2. Run the model training and evaluation for both lakes with given hyperparameters.

### Output:
- Print MSE, Accuracy, F1-score, and AUROC for test sets.
- Display a feature importance bar plot for the top 10 most influential features in the model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score

def train_and_evaluate(lake_name, best_params):
    
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

    # Fill missing values
    input_train.fillna(input_train.mean(), inplace=True)
    input_validate.fillna(input_validate.mean(), inplace=True)
    input_test.fillna(input_test.mean(), inplace=True)

    # Train model using best hyperparameters
    print(f"\nTraining model for {lake_name} with {best_params}...")
    rf_model = RandomForestClassifier(n_estimators=best_params["n_estimators"], 
                                      max_depth=best_params["max_depth"], 
                                      random_state=42)
    rf_model.fit(input_train, output_train.values.ravel())

    # Validate model
    y_validate_pred = rf_model.predict(input_validate)
    mse_validate = mean_squared_error(output_validate, y_validate_pred)

    # Test model
    y_test_pred = rf_model.predict(input_test)
    mse_test = mean_squared_error(output_test, y_test_pred)
    accuracy = accuracy_score(output_test, y_test_pred)
    f1 = f1_score(output_test, y_test_pred, average="weighted")  

    # Calculate AUROC
    y_test_proba = rf_model.predict_proba(input_test)[:, 1]
    auroc = roc_auc_score(output_test, y_test_proba)
    
    print(f"{lake_name} - MSE (Test): {mse_test:.6f}")
    print(f"{lake_name} - Accuracy (Test): {accuracy:.6f}")
    print(f"{lake_name} - F1 Score (Test): {f1:.6f}\n")
    print(f"{lake_name} - AUROC (Test): {auroc:.6f}\n")
    
     # Feature Importance Visualization
    feature_importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': selected_predictors, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance 
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", 
                y="Feature", 
                data=feature_importance_df[:10], 
                hue="Feature", 
                palette="viridis", 
                dodge=False, 
                legend=False) 

    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f'Top 10 Important Features in Random Forest ({lake_name})')
    plt.gca().invert_yaxis()
    plt.show()
    
    return mse_test, accuracy, f1, auroc

if __name__ == "__main__":
    
#     grid search
#     param_grid = {
#         "n_estimators": [50, 100, 200, 250, 300, 350, 400, 500],
#         "max_depth": [5, 10, 15, 20, 25, 30]
#     }

    huron_best_params = {"n_estimators": 350, "max_depth": 25}
    print("\nEvaluating HURON...")
    train_and_evaluate("HURON", huron_best_params)

    simc_best_params = {"n_estimators": 400, "max_depth": 25}
    print("\nEvaluating SIMC...")
    train_and_evaluate("SIMC", simc_best_params)

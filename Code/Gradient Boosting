import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score

def train_gbr_for_lakes(lake_name):
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

    # Best hyperparameters from tuning
    best_params = {
        "HURON": {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 8, "min_samples_split": 100},
        "SIMC": {"learning_rate": 0.1, "n_estimators": 300, "max_depth": 10, "min_samples_split": 50}
    }

    params = best_params[lake_name]
    
    # Train GBR model
    model = GradientBoostingRegressor(
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        loss="squared_error",
        random_state=42
    )
    model.fit(input_train, output_train.values.ravel())
    
    # Predictions
    y_pred = model.predict(input_test)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class labels
    
    # Metrics
    mse = mean_squared_error(output_test, y_pred)
    r2 = r2_score(output_test, y_pred)
    accuracy = accuracy_score(output_test, y_pred_binary)
    f1 = f1_score(output_test, y_pred_binary)
    auroc = roc_auc_score(output_test, y_pred)
    
    print(f"{lake_name} Test Set Evaluation:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"F1 Score: {f1:.6f}")
    print(f"AUROC: {auroc:.6f}")
    
    # Feature Importance Visualization
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': selected_predictors, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df[:10], x='Importance', y='Feature', hue='Feature', dodge=False, palette='viridis')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f'Top 10 Important Features in Gradient Boosting ({lake_name})')
    plt.gca().invert_yaxis()
    plt.show()
    
    return model

if __name__ == "__main__":
    print("\nTraining for HURON...")
    model_huron = train_gbr_for_lakes("HURON")
    
    print("\nTraining for SIMC...")
    model_simc = train_gbr_for_lakes("SIMC")

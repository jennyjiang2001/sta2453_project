pip install torch torchvision torchaudio

"""
This script trains and evaluates a Neural Network model for two different lakes: Huron and Simcoe.

### Steps:
1. Define the function `train_and_evaluate_nn()`:
   - Load selected predictors for each lake.
   - Load training, validation, and test datasets.
   - Handle missing values by filling them with the mean.
   - Normalize features using StandardScaler.
   - Train a fully connected neural network with the given hyperparameters.
   - Evaluate model performance using Mean Squared Error (MSE), Accuracy, F1-score, and AUROC.
   - Visualize feature importance.
2. Run the model training and evaluation for both lakes with the best hyperparameters.

### Output:
- Print MSE, Accuracy, F1-score, and AUROC for test sets.
- Display a feature importance bar plot for the top 10 most influential features in the model.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class PHQModel(nn.Module):
    """
    Neural Network Model with variable hidden layers and sizes.
    """
    def __init__(self, input_size, hidden_num, hidden_size):
        super(PHQModel, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        
        if hidden_num > 0:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            for _ in range(hidden_num - 1):
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
                
            reduced_size = hidden_size // 2
            self.hidden_layers.append(nn.Linear(hidden_size, reduced_size))
            self.output_layer = nn.Linear(reduced_size, 1)
        else:
            self.output_layer = nn.Linear(input_size, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x.view(-1, 1)

def compute_feature_importance(model, predictors):
    """
    Compute feature importance using the first hidden layer's weights.
    """
    first_layer_weights = model.hidden_layers[0].weight.detach().numpy()

    feature_importance = np.mean(np.abs(first_layer_weights), axis=0)

    feature_importance_df = pd.DataFrame({"Feature": predictors, "Importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    return feature_importance_df

def train_and_evaluate_nn(lake_name, best_params, epochs=100, lr=0.001, batch_size=32):
    """
    Trains and evaluates a neural network model for a given lake.
    """
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
    input_train = pd.read_csv(f"{lake_name}_input_train.csv", index_col=0)[predictors[lake_name]]
    input_validate = pd.read_csv(f"{lake_name}_input_validate.csv", index_col=0)[predictors[lake_name]]
    input_test = pd.read_csv(f"{lake_name}_input_test.csv", index_col=0)[predictors[lake_name]]

    output_train = pd.read_csv(f"{lake_name}_output_train.csv")[["Class"]]
    output_validate = pd.read_csv(f"{lake_name}_output_validate.csv")[["Class"]]
    output_test = pd.read_csv(f"{lake_name}_output_test.csv")[["Class"]]

    # Handle missing values
    input_train.fillna(input_train.mean(), inplace=True)
    input_validate.fillna(input_validate.mean(), inplace=True)
    input_test.fillna(input_test.mean(), inplace=True)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(input_train)
    X_validate = scaler.transform(input_validate)
    X_test = scaler.transform(input_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_validate_tensor = torch.tensor(X_validate, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(output_train.values, dtype=torch.float32).view(-1, 1)
    y_validate_tensor = torch.tensor(output_validate.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(output_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    # Initialize model
    print(f"\nTraining Neural Network for {lake_name} with {best_params}...")
    model = PHQModel(input_size=X_train.shape[1], hidden_num=best_params["hidden_num"], hidden_size=best_params["hidden_size"])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    y_test_pred = model(X_test_tensor).detach().numpy().flatten()
    mse_test = mean_squared_error(y_test_tensor.numpy(), y_test_pred)
    accuracy = accuracy_score(np.round(y_test_tensor.numpy()), np.round(y_test_pred))
    f1 = f1_score(np.round(y_test_tensor.numpy()), np.round(y_test_pred), average="weighted")
    auroc = roc_auc_score(y_test_tensor.numpy(), y_test_pred)

    print(f"{lake_name} Test Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean Squared Error: {mse_test:.4f}")
    print(f"AUROC: {auroc:.4f}\n")

    # Compute Feature Importance
    feature_importance_df = compute_feature_importance(model, predictors[lake_name])

    #Plot feature importance
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
    plt.title(f'Top 10 Important Features in Neural Network ({lake_name})')
    plt.gca().invert_yaxis()
    plt.show()
    

if __name__ == "__main__":
    print("\nEvaluating HURON...")
    train_and_evaluate_nn("HURON", {"hidden_num": 2, "hidden_size": 64})

    print("\nEvaluating SIMC...")
    train_and_evaluate_nn("SIMC", {"hidden_num": 2, "hidden_size": 128})

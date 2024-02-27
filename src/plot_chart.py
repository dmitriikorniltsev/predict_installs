import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


def plot_residuals(y_true, y_pred):
    # Calculate residuals
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()

    
def plot_feature_importance(importance_df: pd.DataFrame):
    # Plot feature importance
    plt.figure(figsize=(6, 10))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from Permutation Test')
    plt.gca().invert_yaxis()
    plt.show()
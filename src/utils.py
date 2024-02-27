import json
import os
import yaml

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def read_yaml(path: str) -> dict:
    """Reads yaml file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_json(path: str) -> dict:
    """Reads json file."""
    with open(path, "r") as f:
        return json.load(f)


def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def print_metrics(y_true, y_pred, model_name, path: str = 'data/reports/metrics.json'):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("{} Metrics: \nMAPE: {}\nMSE: {}\nMAE: {}\nR2: {}\n".format(model_name, mape, mse, mae, r2))
    
    # Save metrics to a JSON file
    metrics = {'Model': model_name, 'MAPE': mape, 'MSE': mse, 'MAE': mae, 'R2': r2}
    
    if os.path.exists(path):
        with open(path, 'r+') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]  # convert the dictionary to a list of dictionaries
            data.append(metrics)  # add new metrics dictionary to the list
            f.seek(0)  # reset file position to the beginning
            f.truncate()  # remove existing content
            json.dump(data, f)  # write the updated list back to the file
    else:
        with open(path, 'w') as f:
            json.dump([metrics], f)  # write a new list with the metrics dictionary
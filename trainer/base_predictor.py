import os

import numpy as np
import pandas as pd
import torch

from data_handling.base import BaseDataLoader

"""
This module contains the base implementation of the predictor
class. This class can be used to iterate through a given
test dataset and save the corresponding predictions.
NOTE: This class is intended for testing NOT for validation as
it assumes there are not labels available for the test set.
"""


class BasePredictor:
    def __init__(self, model, test_dataset, n_batch=16):
        self.model = model
        self.test_loader = BaseDataLoader(
            test_dataset, n_batch, 1, False).val_loader
        self.n_test_batch = len(self.test_loader)

    def predict(self, save_path=None):
        """ Implement the prediction logic
        of the test set. In this version it
        saves predictions to a csv file with an id
        column and a prediction column.
        Note:
            Override to output the predictions
            in the format suitable for your tasks.
            Override for specific predictions post-processing
        Args:
            save_path(str): where to save the predictions
        """
        self.model.eval()
        with torch.no_grad():
            predictions = []
            ids = []
            for val_batch in self.test_loader:
                inputs, ids_batch = val_batch["Input"], val_batch["Id"]
                predictions_batch = self.model.predict(inputs.float())
                predictions = np.append(predictions, predictions_batch)
                ids = np.append(ids, ids_batch)
        if save_path is not None:
            predictions_df = pd.DataFrame()
            predictions_df["Id"] = ids
            predictions_df["Prediction"] = predictions
            predictions_df.to_csv(
                os.path.join(save_path, 'predictions.csv'),
                index=False)
        return predictions

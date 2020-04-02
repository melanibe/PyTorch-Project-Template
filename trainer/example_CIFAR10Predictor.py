import os

import numpy as np
import pandas as pd
import torch

from trainer.base_predictor import BasePredictor

"""
An example of an application specific
child class of the BasePredictor class.
"""


class ExampleCIFAR10Predictor(BasePredictor):
    def predict(self, save_path=None):
        """ Implemented the prediction logic
        of the test set.
        Args:
            save_path=(str): where to save the predictions
        """
        onehot_to_labels = self.test_loader.dataset.inverse_transform_label
        self.model.eval()
        with torch.no_grad():
            predictions = []
            ids = []
            for val_batch in self.test_loader:
                inputs, ids_batch = val_batch["Input"], val_batch["Id"]
                onehot_predictions_batch = self.model.predict(inputs.float())
                predictions_batch = onehot_to_labels(onehot_predictions_batch)
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

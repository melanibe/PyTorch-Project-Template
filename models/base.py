import os

import torch
import torch.nn as nn

'''
This module contains the base class to build models.
It implements the save and load methods common to most models.
The init and forward function need to be overridden for your specific model.
See example_simple_net.py for an dummy example.
'''


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self, x):
        """ Implement the model architecture here
        """
        NotImplementedError("The forward function is not implemented yet")

    def predict(self, x):
        """ Implement the prediction logic.
        By default returns the raw output of the network.

        Note:
            Override if necessary (ex: for a classification net
            where raw output corresponds to probabilities instead
            of label predictions.
        """
        return self.forward(x)

    def predict_probas(self, x):
        """ Implement the probability prediction logic
        By default returns the raw output of the network.
        """
        NotImplementedError("The predict_probas function "
                            "is not implemented yet")

    def save(self, save_path, name='model.pt'):
        torch.save(self.state_dict(), os.path.join(save_path, name))

    def restore(self, save_path, name='model.pt'):
        self.load_state_dict(torch.load(os.path.join(save_path, name)))

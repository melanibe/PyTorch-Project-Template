import os

import torch
from torch.utils.tensorboard import SummaryWriter

"""
This module contains a base implementation for a trainer.
It defines a training_step method, a validation_step method
and a train method which consists of the main training loop.
"""


class BaseTrainer:
    def __init__(self, model,
                 criterion, optimizer,
                 data_loader, n_epochs,
                 save_dir,
                 save_model_frequency):
        """
        :param model: the neural model to train
        :param criterion: the loss function to optimize
        :param optimizer: the optimizer to use
        :param data_loader: BaseDataLoader with validation and training loaders
        :param n_epochs: number of epochs to train for
        :param save_dir: directory to use to save tensorboard logs and models
        :param save_model_frequency: frequency (in epochs) to save the model
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = data_loader.train_loader
        if hasattr(data_loader, 'val_loader'):
            self.val_loader = data_loader.val_loader
            self.validate = True
        self.n_epochs = n_epochs
        self.n_train_batch = len(data_loader.train_loader)
        self.n_val_batch = len(data_loader.val_loader)
        self.save_dir = save_dir
        self.writer = SummaryWriter(os.path.join(save_dir, 'summaries'))
        self.save_model_frequency = save_model_frequency

    def train(self):
        for epoch in range(1, self.n_epochs+1):
            running_loss = 0
            for (batch_index, batch) in enumerate(self.train_loader):
                current_loss = self.training_step(batch)
                running_loss += current_loss
                if batch_index % 500 == 499:
                    print("Training loss for epoch {} batch {}: {}"
                          .format(epoch, batch_index, running_loss/500))
                    self.writer.add_scalar(
                        'loss/training',
                        running_loss / 1000,
                        (epoch-1) * self.n_train_batch + batch_index)
                    self.writer.flush()
                    running_loss = 0
            if self.validate:
                validation_loss = self.validation_step()
                print(
                    "Validation loss for epoch {}: {}"
                    .format(epoch, validation_loss))
                self.writer.add_scalar(
                    'loss/validation', validation_loss, epoch)
            if epoch % self.save_model_frequency == 0:
                self.model.save(self.save_dir, 'model-{}.pt'.format(epoch))
        self.writer.flush()
        self.writer.close()

    def training_step(self, batch):
        self.model.train()
        inputs, labels = batch["Input"], batch["Label"]
        # zero the parameter gradients
        self.optimizer.zero_grad()
        outputs = self.model(inputs.float())
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self):
        self.model.eval()
        with torch.no_grad():
            running_val_loss = 0
            for val_batch in self.val_loader:
                inputs, labels = val_batch["Input"], val_batch["Label"]
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()
        val_loss = running_val_loss/self.n_val_batch
        return val_loss

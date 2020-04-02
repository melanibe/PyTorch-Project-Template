# A Simple PyTorch Project Template
This repo contains a code structure that can be used as a simple PyTorch project template. 
It defines classes for dataset creation, data loader creation (including training/validation split handling), model, training procedure definition (including basic tensorboard logging). 

To use this repo for your projects, you can clone/fork this repo and use it as a starting point. To goal is to avoid to rewrite common steps such as
  * `Dataset` to `DataLoader` conversion
  * Validation/training set split logic
  * Training loop definition
  * Model saving and loading procedure
   * ...
 
 every time you start a new project. 
 
 Of course there is no such thing as a one-fits-all code. Hence the idea of the PyTorch project template, is that you derive child classes from the proposed base classes to override some functions according to your specific needs. 
 To show how an example of such "application-specific adaptation" of the project template. I have included some example classes that are derived from the base classes for the [CIFAR10](https://www.kaggle.com/c/cifar-10/) classification task. Note that, these classes are only here to showcase the functionalities of the project template, not to actually achieve any good classification results. The model is the same example model as in the PyTorch classification [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) When you use this project template, you should create your own "example" classes specific to your task, needs and wishes.
 
## Code structure

* `data_handling` folder:
    * `base.py`: 
        This file contains the definition of two base classes derived from PyTorch Dataset and DataLoader classes. `BaseDataset` is here a pure abstract class, all its members need to defined according to the specific shape of the data. See `example_CIFAR10Dataset.py` for an example.
`BaseDataLoader` is derived from `DataLoader`, it can be instantiated as it is. It handles validation/training split of the input dataset in which case it has two attributes: train_loader and val_loader. See `example_main.py` for a usage example in the CIFAR10 classification task.
    * `example_CIFAR10Dataset.py`:
        This file contains an example of an application specific redefinition of the `BaseDataset` class. It is build to transform the data when the data is in form of one folder containing the images where the `id` is the name of the file and labels are stored in a csv. It correspond to the format of the data downloadable from [Kaggle CIFAR10]((https://www.kaggle.com/c/cifar-10/))
    * `utils.py`:
        Use this file to put any data handling helper function you may need.

* `models` folder:
    * `base.py`
        This file contains the base class to build models. It implements the `save()` and `load()` methods common to most models. The `init()` and `forward()` function need to be overridden for your specific model.
        See `example_simple_net.py` for an dummy example.
    * `example_simple_net.py`: dummy example of extension of the `BaseModel` class

* `trainer` folder:
    * `base_trainer.py`
        This module contains a base implementation for a training procedure wrapped in a `BaseTrainer` class. It defines a `training_step()` method, a `validation_step()` method and a `train()` method which consists of the main training loop. Note that the base trainer integrates basic tensorboard support 
(training loss and validation loss are recorded).
    * `base_predictor.py`
        This file contains the base implementation of the `BasePredictor`class. This class can be used to iterate through a given test dataset and save the corresponding predictions.
        NOTE: This class is intended for testing NOT for validation as it assumes there are not labels available for the test set.
    * `example_CIFAR10Predictor.py`
        This file contains an example of an application specific redefinition of the `BasePredictor` class.
        
* `example_main.py` This file contains an very simple example main file to show how to use this PyTorch Project template to build, train, validate, save, load a model
and save predictions to a `.csv` file.



Note: the plan is to further update this template to e.g. handle loading of model parameters via `yaml` config files etc... whenever i'll find the time.


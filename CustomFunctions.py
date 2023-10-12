import os
import scipy
import torch
import sklearn
import xgboost
import catboost
import lightgbm
import numpy as np
import pandas as pd
import librosa as lr
from PIL import Image
import seaborn as sns
import transformers
from typing import Any
import noisereduce as nr
from tqdm.auto import tqdm
import moviepy.editor as mp
from sklearn.svm import SVC
from scipy.io import wavfile
import librosa.display as ld
from nltk.tag import pos_tag
from sklearn.metrics import *
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from datasets import Dataset as dt
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from timeit import default_timer as Timer
from nltk.corpus import stopwords, wordnet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor, CenterCrop, Normalize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier

def performance(model, x_test, y_test):
    """
    Calculates and displays the performance metrics of a trained model.

    Parameters:
    -----------
    model : object
        The trained machine learning model.

    x_test : array-like of shape (n_samples, n_features)
        The input test data.

    y_test : array-like of shape (n_samples,)
        The target test data.

    Returns:
    --------
    None

    Prints:
    -------
    Model Performance:
        Classification report containing precision, recall, F1-score, and support for each class.
    Accuracy:
        The accuracy of the model on the test data.
    Confusion Matrix:
        A plot of the confusion matrix, showing the true and predicted labels for the test data.

    Example:
    --------
    >>> performance(model, x_test, y_test)
    """

    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print("                 Model Performance")
    print(report)
    print(f"Accuracy = {round(accuracy*100, 2)}%")
    matrix = confusion_matrix(y_test, preds)
    matrix_disp = ConfusionMatrixDisplay(matrix)
    matrix_disp.plot(cmap='Reds')
    plt.show()
    
class CustomDataset_CSVlabels(Dataset):
    """
    A PyTorch dataset for loading spectrogram images and their corresponding labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.

    Attributes:
        img_labels (DataFrame): A pandas dataframe containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the image and label at the given index.

    Returns:
        A PyTorch dataset object that can be passed to a DataLoader for batch processing.
    """
    def __init__(self,csv_file, img_dir, transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.img_labels)
    
    def __getitem__(self, index):
        """
        Returns the image and label at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = Image.open(img_path)
        image = image.convert("RGB")
        y_label = torch.tensor(int(self.img_labels.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

def Train_Loop(
        num_epochs:int,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_function:torch.nn.Module,
        device:str
):
    """
    Trains a PyTorch model using the given train and test dataloaders for the specified number of epochs.

    Parameters:
    -----------
    num_epochs : int
        The number of epochs to train the model for.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for the training data.
    test_dataloader : torch.utils.data.DataLoader
        The dataloader for the test/validation data.
    model : torch.nn.Module
        The PyTorch model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to be used during training.
    loss_function : torch.nn.Module
        The loss function to be used during training.

    Returns:
    --------
    None

    Raises:
    -------
    None

    Notes:
    ------
    This function loops over the specified number of epochs and for each epoch, it trains the model on the training
    data and evaluates the performance on the test/validation data. During each epoch, it prints the training loss
    and the test loss and accuracy. At the end of training, it prints the total time taken for training.
    """
    model.to(device)
    start_time = Timer()
    
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch}\n-----------")
        train_loss = 0
        for batch, (x,y) in enumerate(train_dataloader):
            x,y = x.to(device), y.to(device)
            y=y.float().squeeze()
            model.train()
            y_logits = model(x).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            loss = loss_function(y_logits, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch % 10 == 0:
            #     print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)
        
        test_loss, test_acc = 0, 0 
        test_log_loss = 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X,y = X.to(device), y.to(device)
                y = y.float().squeeze()
                test_logits = model(X).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))
                test_loss += loss_function(test_logits, y)
                test_acc += accuracy_score(y_true=y, y_pred=test_pred)
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%\n")

    end_time = Timer()
    print(f"Time taken = {end_time-start_time}")

class CustomDataset_FolderLabels:
    """
    CustomDataset class for loading and splitting a dataset into training, validation, and testing sets.

    Args:
        data_path (str): Path to the main folder containing subfolders for each class.
        train_ratio (float): Ratio of data allocated for the training set (0.0 to 1.0).
        val_ratio (float): Ratio of data allocated for the validation set (0.0 to 1.0).
        test_ratio (float): Ratio of data allocated for the testing set (0.0 to 1.0).
        batch_size (int): Number of samples per batch in the data loaders.
        transform (torchvision.transforms.transforms.Compose): Transformations to be applied on the image

    Attributes:
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): Data loader for the testing set.

    """
    def __init__(self, data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, transform=None):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and splits it into training, validation, and testing sets.

        """
        dataset = ImageFolder(root=self.data_path, transform=self.transform)
        num_samples = len(dataset)

        train_size = int(self.train_ratio * num_samples)
        val_size = int(self.val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def get_train_loader(self):
        """
        Get the data loader for the training set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the training set.

        """
        return self.train_loader

    def get_val_loader(self):
        """
        Get the data loader for the validation set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the validation set.

        """
        return self.val_loader

    def get_test_loader(self):
        """
        Get the data loader for the testing set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the testing set.

        """
        return self.test_loader
    
class CustomDataset_FolderLabelsV2:
    """
    A custom dataset class for loading image data organized in folders with labels.

    Args:
        data_path (str): The root directory path containing subdirectories with images.
        batch_size (int, optional): Batch size for the DataLoader. Default is 32.
        transform (callable, optional): A torchvision.transforms.Compose object for image preprocessing. 
            Default is to resize images to (224, 224), convert them to tensors, and normalize with mean 
            (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).
        shuffle (bool, optional): Whether to shuffle the data during each epoch. Default is True.

    Attributes:
        data_path (str): The path to the root directory containing the dataset.
        batch_size (int): The batch size used for DataLoader.
        transform (callable): The data preprocessing transform applied to each image.
        dataset (torchvision.datasets.ImageFolder): The ImageFolder dataset created from the specified path.
        dataloader (torch.utils.data.DataLoader): The DataLoader for iterating over the dataset.

    Methods:
        get_dataset(): Returns the ImageFolder dataset.
        get_dataloader(): Returns the DataLoader for the dataset.

    Example:
        dataset = CustomDataset_FolderLabelsV2(data_path='dataset_path', batch_size=64)
        dataloader = dataset.get_dataloader()
        for batch in dataloader:
            # Your training loop here.
    """
    def __init__(self, data_path:str, batch_size:int=32, transform=None, shuffle:bool=True):
        self.data_path = data_path
        self.batch_size = batch_size
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.dataset = ImageFolder(root=self.data_path, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle)

    def get_dataset(self):
        return self.dataset
    
    def get_dataloader(self):
        return self.dataloader

def balanced_log_loss(y_true, y_pred, x_test):
    """
    Compute the balanced logarithmic loss.

    Parameters:
        y_true : array-like of shape (n_samples,)
            True class labels.

        y_pred : array-like of shape (n_samples, n_classes)
            Predicted probabilities for each class.

        x_test : array-like of shape (n_samples, n_features)
            Test data used to calculate sample weights.

    Returns:
        balanced_log_loss : float
            The balanced logarithmic loss.

    Notes:
        The balanced logarithmic loss is computed by applying class weights to the
        log loss calculation, where the class weights are based on the distribution
        of class labels in the training data.

        The function first computes sample weights using the 'balanced' strategy,
        which assigns higher weights to minority classes and lower weights to majority
        classes. The sample weights are calculated based on the training data distribution
        of class labels.

        The log loss is then calculated using the true class labels (y_true), predicted
        probabilities (y_pred), and the computed sample weights.

        The balanced logarithmic loss is useful for evaluating classification models
        in imbalanced class scenarios, where each class is roughly equally important
        for the final score.

    Example:
        y_true = [0, 1, 0, 1, 1]
        y_pred = [[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9]]
        x_test = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]

        loss = balanced_log_loss(y_true, y_pred, x_test)
        print(f"Balanced Log Loss: {loss:.5f}")
    """
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_true)
    balanced_log_loss = log_loss(y_true, y_pred, sample_weight=sample_weights)
    return balanced_log_loss

def compare_performance(models:dict, x_test:np.ndarray, y_test:np.ndarray, num_categories:int = None):
    """
    Compare the performance of machine learning models on a test dataset.

    Parameters:
    - models (dict): A dictionary of machine learning models, where the keys are model names and values are the models.
    - x_test (np.ndarray): Test dataset features.
    - y_test (np.ndarray): Test dataset labels.
    - num_categories (int, optional): The number of categories/classes in the dataset. If not provided, it will be determined based on the unique values in y_test.

    Returns:
    - results (pd.DataFrame): A DataFrame containing the performance metrics of the models, including accuracy, precision, recall, and F1-score for each category.

    Note:
    - The models should have a `predict` method compatible with scikit-learn's classifier interface.

    """
    names = []
    accuracy = []
    f1 = {}
    recall = {}
    precision = {}

    if num_categories == None:
      num_categories = len(np.unique(y_test))

    for i in range(num_categories):
       precision[i]= []
       recall[i] = []
       f1[i] = []

    results = pd.DataFrame(columns=['Name', 'Accuracy'])

    for key,value in models.items():
        names.append(key)
        preds = value.predict(x_test)
        f = f1_score(y_test, preds, average=None)
        r = recall_score(y_test, preds, average=None)
        p = precision_score(y_test, preds, average=None)
        accuracy.append(round(accuracy_score(y_test, value.predict(x_test)), 3))

        for i in range(num_categories):
          f1[i].append(round(f[i], 3))
          recall[i].append(round(r[i], 3))
          precision[i].append(round(p[i], 3))

    results['Name'] = names
    results['Accuracy'] = accuracy
    for i in range(num_categories):
      results[f'Precision_{i}'] = precision[i]
       
    for i in range(num_categories):
      results[f'Recall_{i}'] = recall[i]
       
    for i in range(num_categories):
      results[f'f1-score_{i}'] = f1[i]
    
    return results 

LogisticRegression_param_grid = [
    {
    'C': [0.1, 0.2, 0.3, 0.5, 1],
    'penalty': ['l2'],
    'solver' : ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga', 'liblinear', 'saga'],
    'max_iter' : [2000]
    },
    {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1'],
    'solver' : ['liblinear', 'saga'],
    'max_iter' : [2000]
    },
    {
    'C': [0.1, 1, 10, 100],
    'penalty': ['elasticnet'],
    'solver' : ['saga'],
    'l1_ratio': [0.2, 0.4, 0.6],
    'max_iter' : [2000]   
    }
]
  
DecisionTree_param_grid = [
  {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2', None]
  }
]

KNN_param_grid = [
  {
    'n_neighbors': [5,7,10,15],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10,20,30,40,50],
    'metric': ['minkowski', 'cosine'],
    'n_jobs': [-1]
  }
]

SVC_param_grid = [
    {
    'C': [0.1, 0.4, 0.7, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 0.4, 0.7, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'decision_function_shape' : ['ovo', 'ovr']
    }
]

AdaBoost_param_grid = [
  {
    'n_estimators': [30,50,75,100],
    'learning_rate': [0.5,0.75,1,2,5],
    'algorithm': ['SAMME', 'SAMME.R']
  }
]

GradientBoost_param_grid = [
  {
    'loss': ['log_loss', 'deviance', 'exponential'],
    'learning_rate': [0.05,0.1,0.2,0.5,1],
    'n_estimators': [50,75,100,150,200],
    'subsample': [0.4,0.6,0.8,1],
    'criterion': ['friedman_mse', 'squared_error'],
    'max_features': ['sqrt', 'log2', None]
  }
]

RandomForest_param_grid = [
  {
    'n_estimators': [50,75,100,150,200],
    'criterion': ['gini',' entropy', 'log_loss'],
    'max_depth': [5,7,10,None],
    'max_features': ['sqrt', 'log2', None],
    'n_jobs': [-1]
  }
]

HistGradientBoost_param_grid = [
  {
    'loss': ['auto', 'log_loss', 'binary_crossentropy', 'categorical_crossentropy'],
    'learning_rate': [0.05,0.1,0.2,0.5,1],
    'max_iter': [100,200,300,400],
    'l2_regularization': [0,0.2,0.4,0.6,8.0,1]
  }
]

XGBoostClassifier_param_grid = [
  {
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],  
    'colsample_bytree': [0.8, 0.9, 1.0]
  }
]

LGBMClassifier_param_grid = [
  {
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],  
    'colsample_bytree': [0.8, 0.9, 1.0]
  }
]

CatBoostClassifier_param_grid = [
  {
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],  
    'colsample_bytree': [0.8, 0.9, 1.0]
  }
]

def fix_y(y):
  """
    Fix the values in the target variable `y` such that all unique values are in ascending order.

    Parameters:
    -----------
    y : numpy.ndarray or array-like
        The target variable array containing the values to be fixed.

    Returns:
    --------
    numpy.ndarray
        The fixed target variable array with unique values in ascending order.

    Examples:
    ---------
    >>> y = [2, 1, 3, 2, 5, 4, 5, 3]
    >>> fix_y(y)
    array([1, 0, 2, 1, 4, 3, 4, 2])

    >>> y = np.array([3, 5, 2, 1, 4])
    >>> fix_y(y)
    array([2, 3, 1, 0, 2])
    """
  fixed_y = y.copy()
  unique = np.unique(y)
  if unique[0] != 0:
    fixed_y = np.where(fixed_y == unique[0], 0, fixed_y)
    unique[0] = 0
  for i in range(1,len(unique)):
    if unique[i]-unique[i-1] != 1:
      fixed_y = np.where(fixed_y == unique[i], unique[i-1]+1, fixed_y)
      unique[i] = unique[i-1]+1
  return fixed_y

def BestParam_search(models:dict, x, y, num_cores:int = 3):
  """
    Perform hyperparameter tuning using grid search for different models and print the best parameters and scores.

    Parameters:
    -----------
    models : dict
        A dictionary containing model names as keys and the corresponding model objects as values.
    x : array-like
        The feature matrix or input data.
    y : array-like
        The target variable or output data.
    num_cores : int
        Number of cores for parallel processing, default = 3.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> models = {
    ...     'Logistic Regression': LogisticRegression(),
    ...     'Decision Tree': DecisionTreeClassifier(),
    ...     'SVC': SVC()
    ... }
    >>> x = ...
    >>> y = ...
    >>> BestParam_search(models, x, y)
    For Model: Logistic Regression
    Best hyperparameters:  {'C': 1.0, 'penalty': 'l2'}
    Best score:  0.85

    For Model: Decision Tree
    Best hyperparameters:  {'criterion': 'gini', 'max_depth': 5}
    Best score:  0.78

    For Model: SVC
    Best hyperparameters:  {'C': 1.0, 'kernel': 'rbf'}
    Best score:  0.92
    """
  y = fix_y(y)
  for key,model in models.items():
    if isinstance(model, sklearn.linear_model._logistic.LogisticRegression):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, LogisticRegression_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, DecisionTree_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, sklearn.svm._classes.SVC):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, SVC_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, sklearn.neighbors._classification.KNeighborsClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, KNN_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, lightgbm.sklearn.LGBMClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, LGBMClassifier_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, xgboost.sklearn.XGBClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, XGBoostClassifier_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, sklearn.ensemble._weight_boosting.AdaBoostClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, AdaBoost_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, sklearn.ensemble._gb.GradientBoostingClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, GradientBoost_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, sklearn.ensemble._forest.RandomForestClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, RandomForest_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, catboost.core.CatBoostClassifier):
      print(f"Starting Grid Search for Model: {key}")
      grid_search = GridSearchCV(model, CatBoostClassifier_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    elif isinstance(model, sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier):
      grid_search = GridSearchCV(model, HistGradientBoost_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
      print(f"Starting Grid Search for Model: {key}")
      if isinstance(x, scipy.sparse._csr.csr_matrix):
        grid_search.fit(x.toarray(),y)
      else:
         grid_search.fit(x,y)
      print("Search Finished")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)
      print("------------------------")

    else:
      continue

def get_wordnet_pos(treebank_tag:str):
    """
    Map a Treebank part-of-speech tag to the corresponding WordNet part-of-speech tag.

    Parameters:
    - treebank_tag (str): The Treebank part-of-speech tag.

    Returns:
    - str: The corresponding WordNet part-of-speech tag.

    Example:
    >>> get_wordnet_pos('NN')
    'n'

    The function takes a Treebank part-of-speech tag as input and returns the corresponding WordNet
    part-of-speech tag. It can be used to convert part-of-speech tags from Treebank format (used in
    NLTK, for example) to WordNet format.

    The mapping of Treebank tags to WordNet tags is as follows:
    - 'N' (Noun) -> 'n'
    - 'J' (Adjective) -> 'a'
    - 'V' (Verb) -> 'v'
    - 'R' (Adverb) -> 'r'
    - All other cases default to 'n' (Noun).

    Note that the returned WordNet tag is a single character string.

    Please refer to the NLTK documentation for more information about part-of-speech tagging and WordNet.
    """
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class LemmTokenizer:
    """
    Tokenize and lemmatize text using NLTK's WordNetLemmatizer.

    Usage:
    tokenizer = LemmTokenizer()
    tokens = tokenizer("Example sentence")

    The LemmTokenizer class tokenizes and lemmatizes text using NLTK's WordNetLemmatizer. It provides a callable
    object, allowing it to be used as a function for tokenization and lemmatization.

    Methods:
    - __init__(): Initialize the LemmTokenizer object and create an instance of WordNetLemmatizer.
    - __call__(doc): Tokenize and lemmatize the input text.

    Example:
    >>> tokenizer = LemmTokenizer()
    >>> tokens = tokenizer("I am running in the park")
    >>> print(tokens)
    ['I', 'be', 'run', 'in', 'the', 'park']

    Note: This class requires NLTK and its dependencies to be installed.

    Please refer to the NLTK documentation for more information on tokenization, part-of-speech tagging, and lemmatization.
    """

    def __init__(self):
        """
        Initialize the LemmTokenizer object and create an instance of WordNetLemmatizer.
        """
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Tokenize and lemmatize the input text.

        Parameters:
        - doc (str): The text to be tokenized and lemmatized.

        Returns:
        - list: The list of lemmatized tokens.

        The __call__ method tokenizes the input text using word_tokenize and performs part-of-speech tagging using pos_tag.
        It then lemmatizes each token based on its part-of-speech tag using the WordNetLemmatizer and get_wordnet_pos functions.
        The resulting lemmatized tokens are returned as a list.
        """
        tokens = word_tokenize(doc)
        tokens_tags = pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tokens_tags]

def train_MLmodels(models:dict, X_train, y_train):
    """
    Train multiple ML models.

    Parameters:
    - models (list): A list of ML models.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.

    Returns:
    - list: A list of trained models.

    The function trains each model in the input list using the provided training data.
    It iterates over the models, fits each model to the training data, and stores the trained models in a list.
    The list of trained models is returned as the output.
    """
    trained_models = {}

    for key,model in models.items():
        print(f"Training {key}")
        if (isinstance(model, (sklearn.naive_bayes.MultinomialNB, sklearn.naive_bayes.GaussianNB, sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier)) and isinstance(X_train, scipy.sparse._csr.csr_matrix)):
           model.fit(X_train.toarray(), y_train)
        else:
          model.fit(X_train, y_train)
        print(f"{key} Model Trained\n------------------")
        trained_models[key]= model
    return trained_models

def compare_heatmaps(models:dict, x_test:np.ndarray, y_test:np.ndarray):
    """
    Creates a grid of heatmaps comparing the confusion matrices of multiple models.

    Args:
        models (dict): A dictionary of models where the keys represent the model names
            and the values represent the model objects.
        x_test (np.ndarray): The test input data.
        y_test (np.ndarray): The test target data.

    Returns:
        None: Displays the grid of heatmaps comparing the confusion matrices of the models.

    Raises:
        ValueError: If `models` is an empty dictionary.
    """
    if not models:
        raise ValueError("The `models` dictionary must not be empty.")
    data = []
    names = []
    for key, model in models.items():
        data.append(confusion_matrix(y_test, model.predict(x_test)))
        names.append(key)
    num_plots = len(data)
    num_cols = 3
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows*5.5))
    fig.tight_layout(pad=3)

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            sns.heatmap(data[i], cmap="Reds", annot=True, cbar=False, square=True, fmt='d', ax=ax)
            ax.set_title(names[i])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        else:
            fig.delaxes(ax)

    plt.show()


models = {
  'Logistic Regression': LogisticRegression(),
  'Decision Tree': DecisionTreeClassifier(),
  'KNN': KNeighborsClassifier(),
  'Multinomial Naive Bayes': MultinomialNB(),
  'Gaussian Naive Bayes': GaussianNB(),
  'SVC': SVC(),
  'AdaBoost': AdaBoostClassifier(),
  'Gradient Boosting': GradientBoostingClassifier(),
  'Random Forest': RandomForestClassifier(),
  'XGBoost': XGBClassifier(),
  'CatBoost': CatBoostClassifier(),
  'LightGBM': LGBMClassifier()
}

class Train_Classifiers:
    """
        Initialize the Train_Classifiers object.

        Parameters
        ----------
        x : array-like
            The input features for training the models.
        y : array-like
            The target variable for training the models.
        models : dict, optional
            A dictionary containing the models to train. The keys represent the model names, and the values are
            the model objects. Defaults to `models`.
        test_size : int, optional
            The proportion of the dataset to use as the test set. Defaults to 0.2.
    """
    def __init__(self, x, y, models:dict=models, test_size:int=0.2) -> None:
        self.x = x
        self.y = fix_y(y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, stratify=self.y)
        self.models = models

    def resample_trainData(self, strategy='auto', neighbors:int=5):
        """
        Resamples the training data using the Synthetic Minority Over-sampling Technique (SMOTE).

        This method applies the Synthetic Minority Over-sampling Technique (SMOTE) to balance
        the class distribution in the training data. It generates synthetic samples for the
        minority class to address class imbalance problems in classification tasks.

        Parameters:
        - strategy (str, optional): The sampling strategy to be used by SMOTE. It can take on
        values like 'auto', 'minority', 'not minority', 'all', etc. Default is 'auto'.
        - neighbors (int, optional): The number of nearest neighbors to be considered when
        generating synthetic samples. It affects the degree of similarity between the
        generated samples and the existing ones. Default is 5.

        Note:
        The `x_train` and `y_train` attributes of the object are updated with the resampled data.

        Example:
        >>> model = YourClassifier()
        >>> model.fit(x_train, y_train)
        >>> model.resample_trainData(strategy='minority', neighbors=10)

        """
        self.x_train, self.y_train = SMOTE(sampling_strategy=strategy, k_neighbors=neighbors, random_state=42).fit_resample(self.x_train, self.y_train)
    
    def fit(self):
        """
        Train the models using the provided training data.
        """
        self.trained_models = {}
        for key, model in self.models.items():
            print(f"Training {key}")
            start = Timer()
            try:
                model.fit(self.x_train, self.y_train)
            except TypeError:
                model.fit(self.x_train.toarray(), self.y_train)
            end = Timer()
            print(f"{key} Model Trained\nTime taken = {round(end-start, 3)} seconds\n------------------")
            self.trained_models[key]= model
    
    def score(self):
        """
        Evaluate the performance of the trained models on the test set.
        """
        self.predicts = {}
        for key, model in self.trained_models.items():
            try:
                self.predicts[key] = model.predict(self.x_test)
                print(f"{key}: {accuracy_score(self.y_test, self.predicts[key])}")
            except TypeError:
                self.predicts[key] = model.predict(self.x_test.toarray())
                print(f"{key}: {accuracy_score(self.y_test, self.predicts[key])}")

    def get_trained_models(self):
        """
        Get the dictionary of trained models.

        Returns
        -------
        dict
            A dictionary containing the trained models.
        """
        return self.trained_models
    
    def get_single_model(self,key):
        """
        Get a specific trained model.

        Parameters
        ----------
        key : str
            The key representing the model.

        Returns
        -------
        object
            The trained model.
        """
        return self.trained_models[key]
    
    def model_performance(self, key):
        """
        Calculate and display the performance metrics of a specific model.

        Parameters
        ----------
        key : str
            The key representing the model.
        Returns:
        --------
        None

        Prints:
        -------
        Model Performance:
            Classification report containing precision, recall, F1-score, and support for each class.
        Accuracy:
            The accuracy of the model on the test data.
        Confusion Matrix:
            A plot of the confusion matrix, showing the true and predicted labels for the test data.
        """
        accuracy = accuracy_score(self.y_test, self.predicts[key])
        report = classification_report(self.y_test, self.predicts[key])
        print("\t\t\t\t\tModel Performance")
        print(report)
        print(f"Accuracy = {round(accuracy*100, 2)}%")
        matrix = confusion_matrix(self.y_test, self.predicts[key])
        matrix_disp = ConfusionMatrixDisplay(matrix)
        matrix_disp.plot(cmap='Blues')
        plt.show()

    def Compare_ConfusionMatrices(self):
        """
        Creates a grid of heatmaps comparing the confusion matrices of multiple models.

        Returns:
            None: Displays the grid of heatmaps comparing the confusion matrices of the models.
        """
        data = []
        names = []
        for key, model in self.trained_models.items():
            data.append(confusion_matrix(self.y_test, self.predicts[key]))
            names.append(key)
        num_plots = len(data)
        num_cols = 3
        num_rows = (num_plots - 1) // num_cols + 1
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows*5.5))
        fig.tight_layout(pad=3)

        for i, ax in enumerate(axes.flat):
            if i < num_plots:
                sns.heatmap(data[i], cmap="Reds", annot=True, cbar=False, square=True, fmt='d', ax=ax)
                ax.set_title(names[i])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            else:
                fig.delaxes(ax)

        plt.show()

    def Compare_Performance(self):
        """
        Compare the performance metrics (accuracy, precision, recall, f1-score) of all trained models.

        Returns:
        - results (pd.DataFrame): A DataFrame containing the performance metrics of the models, including accuracy, AUC, precision, recall, and F1-score for each category.

        Note:
        - The models should have a `predict` method compatible with scikit-learn's classifier interface.

        """
        names = []
        accuracy = []
        f1 = {}
        recall = {}
        precision = {}
        roc = []

        self.probabilites = {}

        num_categories = len(np.unique(self.y_test))

        for i in range(num_categories):
            precision[i]= []
            recall[i] = []
            f1[i] = []

        self.results = pd.DataFrame(columns=['Name', 'Accuracy'])

        for key,value in self.trained_models.items():
            names.append(key)
            f = f1_score(self.y_test, self.predicts[key], average=None)
            r = recall_score(self.y_test, self.predicts[key], average=None)
            p = precision_score(self.y_test, self.predicts[key], average=None)
            accuracy.append(round(accuracy_score(self.y_test, self.predicts[key]), 3))

            if self.y_test.shape[0] > 2:
                try:
                    self.probabilites[key] = value.predict_proba(self.x_test)
                    roc.append(round(roc_auc_score(self.y_test, self.probabilites[key], multi_class='ovr'), 3))
                except AttributeError:      # forgot to set probability = True while creating SVC model
                    self.probabilites[key] = None
                    roc.append(None)
                except TypeError:           # passed scipy sparse array into models requiring dense data
                    self.probabilites[key] = value.predict_proba(self.x_test.toarray())
                    roc.append(round(roc_auc_score(self.y_test, self.probabilites[key], multi_class='ovr'), 3))
            else:
                try:
                    self.probabilites[key] = value.predict_proba(self.x_test)[:,1]
                    roc.append(round(roc_auc_score(self.y_test, self.probabilites[key]), 3))
                except AttributeError:
                    self.probabilites[key] = None
                    roc.append(None)
                except TypeError:
                    self.probabilites[key] = value.predict_proba(self.x_test.toarray())[:,1]
                    roc.append(round(roc_auc_score(self.y_test, self.probabilites[key]), 3))

            for i in range(num_categories):
                f1[i].append(round(f[i], 3))
                recall[i].append(round(r[i], 3))
                precision[i].append(round(p[i], 3))

        self.results['Name'] = names
        self.results['Accuracy'] = accuracy
        self.results['AUC'] = roc
        for i in range(num_categories):
            self.results[f'Precision_{i}'] = precision[i]
        
        for i in range(num_categories):
            self.results[f'Recall_{i}'] = recall[i]
        
        for i in range(num_categories):
            self.results[f'f1-score_{i}'] = f1[i]
        
        return self.results

    def Best_ParamSearch(self, num_cores:int = 1):
        """
        Perform a grid search to find the best hyperparameters for each model in the class.

        Parameters
        ----------
        num_cores : int, optional
            The number of CPU cores to use during the grid search. Defaults to 1.

        Notes
        -----
        This method performs a grid search with cross-validation for each model in the class.
        It searches for the best hyperparameters using the provided `model_param_grid` dictionaries
        specific to each model type. The best hyperparameters and corresponding best scores are
        printed for each model.

        For the models with sparse input (`HistGradientBoostingClassifier` and certain Naive Bayes models),
        the grid search is performed on the dense representation of the input data if it is initially sparse.

        The printed information includes the model name, the start and finish of the grid search, the
        best hyperparameters found, and the corresponding best score.

        Models that are not recognized or supported will be skipped.

        Examples
        --------
        # Perform a grid search for the best hyperparameters
        classifier = Train_Classifiers(x_train, y_train, models)
        classifier.Best_ParamSearch(num_cores=2)
        """
        for key,model in self.models.items():
            if isinstance(model, sklearn.linear_model._logistic.LogisticRegression):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, LogisticRegression_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, DecisionTree_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, sklearn.svm._classes.SVC):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, SVC_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, sklearn.neighbors._classification.KNeighborsClassifier):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, KNN_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, lightgbm.sklearn.LGBMClassifier):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, LGBMClassifier_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, xgboost.sklearn.XGBClassifier):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, XGBoostClassifier_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, sklearn.ensemble._weight_boosting.AdaBoostClassifier):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, AdaBoost_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, sklearn.ensemble._gb.GradientBoostingClassifier):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, GradientBoost_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, sklearn.ensemble._forest.RandomForestClassifier):
                print(f"Started Grid Search for Model: {key}")
                grid_search = GridSearchCV(model, RandomForest_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            elif isinstance(model, (sklearn.naive_bayes.MultinomialNB, sklearn.naive_bayes.GaussianNB, sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier)):
                grid_search = GridSearchCV(model, HistGradientBoost_param_grid, cv=5, scoring='accuracy', n_jobs=num_cores)
                print(f"Started Grid Search for Model: {key}")
                if isinstance(self.x,self. scipy.sparse._csr.csr_matrix):
                    grid_search.fit(self.x.toarray(),self.y)
                else:
                    grid_search.fit(self.x,self.y)
                print("Search Finished")
                print("Best hyperparameters: ", grid_search.best_params_)
                print("Best score: ", grid_search.best_score_)
                print("------------------------")

            else:
                continue



def tfidf_preprocessing(x, y, encoder=None, remove_stopwords:bool =True, test_size:float = 0.2, stratify=False, shuffle:bool=True):
    """
    Preprocesses the input data using TF-IDF vectorization.

    Args:
        x (list or array-like): Input data.
        y (list or array-like): Target labels.
        encoder (optional): Encoder for the target labels. Default is None.
        remove_stopwords (bool, optional): Flag indicating whether to remove stopwords. Default is True.
        test_size (float, optional): The proportion of the data to use for testing. Default is 0.2.
        stratify (bool, optional): Flag indicating whether to perform stratified splitting. Default is False.

    Returns:
        tuple: A tuple containing the preprocessed data and labels:
            - x_train (sparse matrix): Training data after TF-IDF vectorization.
            - x_test (sparse matrix): Testing data after TF-IDF vectorization.
            - y_train (array-like): Training labels.
            - y_test (array-like): Testing labels.

    Raises:
        TypeError: If `remove_stopwords` is not a boolean value.
        ValueError: If `test_size` is not between 0 and 1 (exclusive).
    """
    if not isinstance(remove_stopwords, bool):
        raise TypeError("Invalid input! Expected a boolean value.")
    
    if not (0 < test_size < 1):
        raise ValueError("Invalid input! The value must be between 0 and 1 (exclusive).")
    
    if encoder is not None:
        y = encoder.fit_transform(y)
    
    if remove_stopwords == True:
        stop_words = set(stopwords.words('english'))
        for i, a in enumerate(x):
            try:
                tokens = word_tokenize(a)
                filtered = [word for word in tokens if word not in stop_words]
                x[i] = ' '.join(filtered)
            except TypeError:
                continue

    if stratify==True:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, shuffle=shuffle, stratify=y)
    else:        
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, shuffle=shuffle)

    vectorizer = TfidfVectorizer(tokenizer=LemmTokenizer(), sublinear_tf=True)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    
    return x_train, x_test, y_train, y_test

class BERT_Embeddings:
    """
    A class for creating BERT embeddings from a pandas DataFrame.

    Parameters:
        data (pd.core.frame.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the column in the DataFrame containing the target variable.
        test_size (float, optional): The proportion of data to use for the test set. Default is 0.2.
        model (str, optional): The BERT model to use. Default is 'distilbert-base-uncased'.
        stratify (bool, optional): Whether to stratify the train-test split based on the target_column. Default is True.

    Attributes:
        data (dt.Frame): A datatable Frame converted from the input DataFrame.
        target_column (str): The name of the target column.
        model (transformers.BertModel or transformers.DistilBertModel): The BERT model used for embedding extraction.
        tokenizer (transformers.BertTokenizer or transformers.DistilBertTokenizer): The tokenizer for the selected BERT model.
        encoded_data (dt.Frame): The tokenized and encoded data.
        hidden_states (dt.Frame): The extracted hidden states from the BERT model.
        x_train (np.ndarray): Numpy array containing the hidden states for the training set.
        y_train (np.ndarray): Numpy array containing the target values for the training set.
        x_test (np.ndarray): Numpy array containing the hidden states for the test set.
        y_test (np.ndarray): Numpy array containing the target values for the test set.

    Methods:
        extract_hidden_states(batch): Extracts hidden states from a batch using the BERT model.
        encode_data(column): Tokenizes and encodes the data for the specified column.
        create_hidden_states(batch_size): Creates hidden states for the encoded data in batches.
        create_train_test_set(): Splits the hidden states and target values into train and test sets.
        get_train_test(): Returns the training and test sets (x_train, x_test, y_train, y_test).
        get_all_data(): Returns all data, encoded data, and hidden states.

    """
    def __init__(self, data:pd.core.frame.DataFrame, target_column:str,test_size:float = 0.2, model:str = 'distilbert-base-uncased', stratify:bool=True, shuffle:bool=True) -> None:
        self.data = dt.from_pandas(data)
        self.target_column = target_column
        if "__index_level_0__" in self.data.column_names:
          self.data.remove_columns(["__index_level_0__"])

        if stratify == True:
          self.data = self.data.train_test_split(test_size=test_size, shuffle=shuffle, stratify_by_column=self.target_column)
        else:
          self.data = self.data.train_test_split(test_size=test_size, shuffle=shuffle)

        if "distil" in model:
            self.model = transformers.DistilBertModel.from_pretrained(model)
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(model)
        else:
            self.model = transformers.BertModel.from_pretrained(model)
            self.tokenizer = transformers.BertTokenizer.from_pretrained(model)
    
    def extract_hidden_states(self,batch):
        """
        Extracts hidden states from a batch using the BERT model.

        Parameters:
            batch: A batch of tokenized and encoded data.

        Returns:
            dict: A dictionary containing the extracted hidden states for the batch.
        """
        inputs = {k:v.to('cpu') for k, v in batch.items() if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
    
    def encode_data(self, column):
        """
        Tokenizes and encodes the data for the specified column.

        Parameters:
            column (str): The column name for which to tokenize and encode the data.
        """
        def tokenize(batch):
          return self.tokenizer(batch[column], padding=True, truncation=True)
        self.encoded_data = self.data.map(tokenize, batched=True, batch_size=None)

    def create_hidden_states(self, batch_size:int = 500):
        """
        Creates hidden states for the encoded data in batches.

        Parameters:
            batch_size (int, optional): The batch size for processing the encoded data. Default is 500.
        """
        self.encoded_data.set_format("torch", columns=["input_ids", "attention_mask", self.target_column])
        self.hidden_states = self.encoded_data.map(self.extract_hidden_states, batched=True, batch_size=batch_size)

    def create_train_test_set(self):
        """
        Splits the hidden states and target values into train and test sets.
        """
        self.x_train = np.array(self.hidden_states['train']['hidden_state'])
        self.y_train = np.array(self.hidden_states['train'][self.target_column])
        self.x_test = np.array(self.hidden_states['test']['hidden_state'])
        self.y_test = np.array(self.hidden_states['test'][self.target_column])

    def get_train_test(self):
        """
        Returns the training and test sets.

        Returns:
            tuple: A tuple containing numpy arrays (x_train, x_test, y_train, y_test) representing the training and test data.
        """
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_all_data(self):
        """
        Returns all data, encoded data, and hidden states.

        Returns:
            tuple: A tuple containing datatable Frames representing the data, encoded data, and hidden states.
        """
        return self.data, self.encoded_data, self.hidden_states
    
def CrossValidateClassifiers(models: dict, x, y, cv: int = 5, scoring: str = 'accuracy', return_performance: bool = False):
    """
    Cross-validate multiple classifiers and evaluate their performance.

    Args:
        models (dict): A dictionary containing classifier names as keys and classifier objects as values.
        x (array-like): Input features for classification.
        y (array-like): Target labels.
        cv (int, optional): Number of cross-validation folds. Default is 5.
        scoring (str, optional): Scoring metric for evaluation. Default is 'accuracy'.
        return_performance (bool, optional): If True, returns a dictionary of performance metrics for each classifier. Default is False.

    Returns:
        dict or None: A dictionary of performance metrics for each classifier if return_performance is True, else None.

    Note:
        This function performs cross-validation for each classifier in the provided models dictionary,
        and it prints the test scores, mean test score, and time taken for each classifier.

    Example:
        models = {'Random Forest': RandomForestClassifier(), 'Logistic Regression': LogisticRegression()}
        x_train, y_train = load_data()
        CrossValidateClassifiers(models, x_train, y_train, cv=10, scoring='f1', return_performance=True)
    """
    model_performance = {}
    for key, model in models.items():
        start = Timer()
        try:
            model_performance[key] = cross_validate(model, x, y, scoring=scoring, cv=cv, n_jobs=3)
        except ValueError:
            model_performance[key] = cross_validate(model, MinMaxScaler().fit_transform(x), y, scoring=scoring, cv=cv, n_jobs=3)
        end = Timer()
        print("For {}\nTest Scores = {}\nMean Test Score = {}\nTime taken = {} seconds\n".format(key, model_performance[key]['test_score'], model_performance[key]['test_score'].mean(), round(end-start, 3)))

    if return_performance:
        return model_performance

EfficientNet_transform = Compose([
    ToTensor(),
    Resize((384,384), interpolation=InterpolationMode.BICUBIC),
    CenterCrop((384,384)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
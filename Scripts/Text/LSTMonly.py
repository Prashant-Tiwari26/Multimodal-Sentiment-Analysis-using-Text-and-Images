import sys
sys.path.append("C:\College\Projects\Multimodal Sentiment Analysis using Text and Images")

import numpy as np
import pandas as pd
from Utils.models import LSTMmodel
from Utils.nlp import clean_caption
from nltk import word_tokenize
from Utils.neural_net import TrainLoop
from torch.utils.data import Dataset, DataLoader

y = np.load("Data/Text/TF-IDF/labels.npy")
data = pd.read_csv("Data/Text/Engineered.csv")
data['Caption'] = data['Caption'].apply(clean_caption)
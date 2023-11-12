import sys
sys.path.append("C:\College\Projects\Multimodal Sentiment Analysis using Text and Images")
from Utils.nlp import get_word_to_index
from Utils.models import MultimodalImageStatic
from Utils.neural_net import TrainLoopCompact, MultimodalDatasetStatic

import ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

data = pd.read_csv("Data/multimodal.csv", index_col=False)
data['Caption'] = data['Caption'].apply(ast.literal_eval)

image_embeddings = np.load("Data/Images/Image Embeddings/EfficientNet.npy")

train, test, train_image, test_image = train_test_split(data, image_embeddings, test_size=0.2, shuffle=True, stratify=data['LABEL'], random_state=1)
train, val, train_image, val_image = train_test_split(train, train_image, test_size=0.125, shuffle=True, stratify=train['LABEL'], random_state=1)

word_to_index = get_word_to_index("glove.twitter.27B/glove.twitter.27B.25d.txt")

train_text = np.array(train['Caption'])
val_text = np.array(val['Caption'])

train_label = np.array(train['LABEL'])
val_label = np.array(val['LABEL'])

train_set = MultimodalDatasetStatic(train_text, train_image, train_label, word_to_index)
val_set = MultimodalDatasetStatic(val_text, val_image, val_label, word_to_index)

train_loader = DataLoader(train_set, 32)
val_loader = DataLoader(val_set, 32)

model = MultimodalImageStatic(3, [512, 256], 1280, 25, 256, bidirectionality=True, freeze=False)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
loss_fun = torch.nn.NLLLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', 0.4, 8)
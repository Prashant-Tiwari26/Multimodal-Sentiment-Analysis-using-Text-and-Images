import sys
sys.path.append("C:\College\Projects\Multimodal Sentiment Analysis using Text and Images")
from Utils.models import MultimodalModel
from Utils.nlp import get_word_to_index
from Utils.neural_net import TrainLoopMultimodal, MultimodalDataset

import ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.efficientnet import EfficientNet_B1_Weights

def MultimodalTrain():
    data = pd.read_csv("Data/multimodal.csv", index_col=False)
    data['Caption'] = data['Caption'].apply(ast.literal_eval)

    train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['LABEL'], random_state=1)
    train, val = train_test_split(train, test_size=0.125, shuffle=True, stratify=train['LABEL'], random_state=1)

    word_to_index = get_word_to_index("glove.twitter.27B/glove.twitter.27B.25d.txt")

    train_set = MultimodalDataset(np.array(train['Caption']), "Data/Images", np.array(train['LABEL']), np.array(train['File Name']), word_to_index, EfficientNet_B1_Weights.IMAGENET1K_V2.transforms())
    val_set = MultimodalDataset(np.array(val['Caption']), "Data/Images", np.array(val['LABEL']), np.array(val['File Name']), word_to_index, EfficientNet_B1_Weights.IMAGENET1K_V2.transforms())

    train_loader = DataLoader(train_set, 32)
    val_loader = DataLoader(val_set, 32)

    model_1 = MultimodalModel(4, 25, 256, [512], bidirectionality=True, weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
    optimizer = torch.optim.NAdam(model_1.parameters(), lr=0.001)
    loss_fun = torch.nn.NLLLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', 0.4, 8)

    TrainLoopMultimodal(model_1, optimizer, loss_fun, train_loader, val_loader, scheduler, 100, 20, True, 'cpu')
    torch.save(model_1.state_dict(), 'mm1.pth')

if __name__ == '__main__':
    MultimodalTrain()
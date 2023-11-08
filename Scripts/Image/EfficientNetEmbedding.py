import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights

import sys
sys.path.append("C:\College\Projects\Multimodal Sentiment Analysis using Text and Images")
from Utils.CustomFunctions import CustomDataset_CSVlabels, EfficientNet_transform

def GenerateEmbeddings():
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])

    dataset = CustomDataset_CSVlabels("Data/Images/ImageLabelsSequenced.csv", "Data/Images", transform=EfficientNet_transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    embeddings = []
    cat = []
    model.to('cuda')
    model.eval()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        embeddings.extend(outputs.cpu().cpu().detach().numpy())
        cat.extend(labels.cpu().detach().numpy())

    embeddings = np.array(embeddings)
    np.save("Data/ImageEmbeddings/EfficientNet.npy")

if __name__ == '__main__':
    GenerateEmbeddings()
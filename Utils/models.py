import torch
import torchtext
from torchvision.models.efficientnet import efficientnet_b1, EfficientNet_B1_Weights

class LSTMmodel(torch.nn.Module):
    def __init__(self, n_layers:int, embed_dim:int, hidden_dim:int, embedding:str="twitter.27B", bidirectionality:bool=False, freeze:bool=False) -> None:
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embedding, embed_dim)
        if not bidirectionality:
            self.model = torch.nn.Sequential(
                torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
                torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True),
                torch.nn.Linear(hidden_dim, 3)
            )
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
                torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True),
                torch.nn.Linear(2 * hidden_dim, 3)
            )

    def forward(self, x):
        return torch.nn.functional.log_softmax(self.model(x), dim=1)
    
class MultimodalModel(torch.nn.Module):
    def __init__(self, n_layers:int, embed_dim:int, hidden_dim:int, neurons:list, embedding:str="twitter.27B", bidirectionality:bool=False, freeze:bool=False, weights=None) -> None:
        super().__init__()
        model = efficientnet_b1(weights=weights)
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2,True))
        self.CNN = model

        glove_embeddings = torchtext.vocab.GloVe(embedding, embed_dim)
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality),
        )

        self.linear_layers = torch.nn.ModuleList()
        if bidirectionality == True:
            self.linear_layers.append(torch.nn.Linear(1280+(2*hidden_dim), neurons[0]))
        else:
            self.linear_layers.append(torch.nn.Linear(1280+hidden_dim, neurons[0]))
        self.linear_layers.append(torch.nn.SELU())
        self.linear_layers.append(torch.nn.Dropout(0.3))
        for i in range(1, len(neurons)):
            self.linear_layers.append(torch.nn.Linear(neurons[i-1], neurons[i]))
            self.linear_layers.append(torch.nn.BatchNorm1d(neurons[i]))
            self.linear_layers.append(torch.nn.SELU())
        self.linear_layers.append(torch.nn.Dropout(0.3))
        self.linear_layers.append(torch.nn.Linear(neurons[-1], 3))

    def forward(self, text, image):
        image_embeddings = self.CNN(image)
        text_output, _ = self.LSTM(text)  
        text_embeddings = text_output[:, -1, :]
        multimodal = torch.cat([image_embeddings, text_embeddings], dim=1)
        for layer in self.linear_layers:
            multimodal = layer(multimodal)
        return torch.nn.functional.log_softmax(multimodal, dim=1)
    
class MultimodalImageStatic(torch.nn.Module):
    def __init__(self, n_layers:int, linear_neurons:list, image_dim:int, embed_dim:int, hidden_dim:int=256, embedding:str="twitter.27B", bidirectionality:bool=False, freeze:bool=False):
        glove_embeddings = torchtext.vocab.GloVe(embedding, embed_dim)
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality),
        )
        self.linear_layers = torch.nn.ModuleList()
        if bidirectionality == True:
            self.linear_layers.append(torch.nn.Linear(image_dim+(2*hidden_dim), linear_neurons[0]))
        else:
            self.linear_layers.append(torch.nn.Linear(image_dim+hidden_dim, linear_neurons[0]))
        self.linear_layers.append(torch.nn.SELU())
        self.linear_layers.append(torch.nn.Dropout(0.3))
        for i in range(1, len(linear_neurons)):
            self.linear_layers.append(torch.nn.Linear(linear_neurons[i-1], linear_neurons[i]))
            self.linear_layers.append(torch.nn.BatchNorm1d(linear_neurons[i]))
            self.linear_layers.append(torch.nn.SELU())
        self.linear_layers.append(torch.nn.Dropout(0.3))
        self.linear_layers.append(torch.nn.Linear(linear_neurons[-1], 3))

    def forward(self, text, image_embeddings):
        text_output, _ = self.LSTM(text)  
        text_embeddings = text_output[:, -1, :]
        multimodal = torch.cat([image_embeddings, text_embeddings], dim=1)
        for layer in self.linear_layers:
            multimodal = layer(multimodal)
        return torch.nn.functional.log_softmax(multimodal, dim=1)
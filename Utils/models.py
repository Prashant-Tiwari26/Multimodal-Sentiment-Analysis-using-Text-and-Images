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
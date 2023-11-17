import torch
import torchtext
from torchvision.models.efficientnet import efficientnet_b1, EfficientNet_B1_Weights

class LSTMmodel(torch.nn.Module):
    def __init__(self, n_layers:int, embed_dim:int, hidden_dim:int, embedding:str="twitter.27B", bidirectionality:bool=False, freeze:bool=False) -> None:
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embedding, embed_dim)
        
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality),
        )
        self.linear = torch.nn.LazyLinear(3)

    def forward(self, x):
        a, _ = self.LSTM(x)
        a = a[:,-1,:]
        return torch.nn.functional.log_softmax(self.linear(a) , dim=1)
    
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
        self.linear_layers.append(torch.nn.LazyLinear(neurons[0]))
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
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embedding, embed_dim)
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality),
        )
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.LazyLinear(linear_neurons[0]))
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
    
class MultimodalSelfAttention(torch.nn.Module):
    def __init__(self, n_layers:int, linear_neurons:list, image_dim:int, embed_dim:int, hidden_dim:int=256, embedding:str="twitter.27B", bidirectionality:bool=False, freeze:bool=False):
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embedding, embed_dim)
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality),
        )
        if bidirectionality == False:
            self.Attention = torch.nn.MultiheadAttention(image_dim+hidden_dim, 8)
        else:
            self.Attention = torch.nn.MultiheadAttention(image_dim+(2*hidden_dim), 8)
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.LazyLinear(linear_neurons[0]))
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
        text_embeddings = text_embeddings.unsqueeze(0).permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        image_embeddings = image_embeddings.unsqueeze(0).permute(1, 0, 2)  # (seq_len, batch, image_dim)
        combined = torch.cat([text_embeddings, image_embeddings], dim=2)  # (seq_len, batch, embed_dim + image_dim)
        attended, _ = self.Attention(combined, combined, combined)
        attended = attended[:, -1, :]  # Considering only the last sequence output
        
        for layer in self.linear_layers:
            attended = layer(attended)
        
        return torch.nn.functional.log_softmax(attended, dim=1)
    
class MultimodalSelfAttentionv2(torch.nn.Module):
    def __init__(self, n_layers:int, linear_neurons:list, n_heads:list, image_dim:int, embed_dim:int, hidden_dim:int=256, embedding:str="twitter.27B", bidirectionality:bool=False, freeze:bool=False):
        super().__init__()
        glove_embeddings = torchtext.vocab.GloVe(embedding, embed_dim)
        self.LSTM = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(glove_embeddings.vectors, freeze=freeze),
            torch.nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectionality),
        )
        self.Attention_layers = torch.nn.ModuleList()
        if bidirectionality == False:
            self.Attention_layers.append(torch.nn.MultiheadAttention(image_dim+hidden_dim, n_heads[0]))
            self.Attention_layers.append(torch.nn.SELU())
            for i in range(1,len(n_heads)):
                self.Attention_layers.append(torch.nn.MultiheadAttention(image_dim+hidden_dim, n_heads[i]))
                self.Attention_layers.append(torch.nn.SELU())
        else:
            self.Attention_layers.append(torch.nn.MultiheadAttention(image_dim+(2*hidden_dim), n_heads[0]))
            self.Attention_layers.append(torch.nn.SELU())
            for i in range(1,len(n_heads)):
                self.Attention_layers.append(torch.nn.MultiheadAttention(image_dim+(2*hidden_dim), n_heads[i]))
                self.Attention_layers.append(torch.nn.SELU())
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.LazyLinear(linear_neurons[0]))
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
        text_embeddings = text_embeddings.unsqueeze(0).permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        image_embeddings = image_embeddings.unsqueeze(0).permute(1, 0, 2)  # (seq_len, batch, image_dim)
        combined = torch.cat([text_embeddings, image_embeddings], dim=2)  # (seq_len, batch, embed_dim + image_dim)
        attended, _ = self.Attention(combined, combined, combined)
        attended = attended[:, -1, :]  # Considering only the last sequence output
        
        for layer in self.linear_layers:
            attended = layer(attended)
        
        return torch.nn.functional.log_softmax(attended, dim=1)
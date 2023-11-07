import torch
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose, ToTensor, InterpolationMode, CenterCrop, Normalize, Resize

class CustomTextDataset(Dataset):
    def __init__(self, data, word_to_index, labels, max_seq_length):
        self.data = data
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        text_indices = [self.word_to_index.get(word, 0) for word in text]
        
        return {
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

EfficientNet_transform = Compose([
    ToTensor(),
    Resize((384,384), interpolation=InterpolationMode.BICUBIC),
    CenterCrop((384,384)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def TrainLoop(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    num_epochs:int=20,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    device:str='cpu'
):
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    total_train_loss = []
    total_val_loss = []
    best_model_weights = model.state_dict()

    train_accuracies = []
    val_accuracies = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("\nEpoch {}\n----------".format(epoch))
        train_loss = 0
        for i, batch in enumerate(train_dataloader):
            text_indices = batch['text_indices'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(text_indices)
            loss = criterion(outputs, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()
            print("Loss for batch {} = {}".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            val_true_labels = []
            train_true_labels = []
            val_pred_labels = []
            train_pred_labels = []
            for batch in val_dataloader:
                text_indices = batch['text_indices'].to(device)
                labels = batch['label'].to(device)
                outputs = model(text_indices)
                loss = criterion(outputs, labels)
                validation_loss += loss

                outputs = torch.argmax(outputs, dim=1)
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(outputs.cpu().numpy())

            for batch in train_dataloader:
                text_indices = batch['text_indices'].to(device)
                labels = batch['label'].to(device)
                outputs = model(text_indices)

                outputs = torch.argmax(outputs, dim=1)
                train_true_labels.extend(labels.cpu().numpy())
                train_pred_labels.extend(outputs.cpu().numpy())

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            val_true_labels = np.array(val_true_labels)
            train_true_labels = np.array(train_true_labels)
            val_pred_labels = np.array(val_pred_labels)
            train_pred_labels = np.array(train_pred_labels)

            train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
            val_accuracy = accuracy_score(val_true_labels, val_pred_labels)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")

            print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if epochs_without_improvement == early_stopping_rounds:
            break

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))

    if return_best_model == True:
        model.load_state_dict(best_model_weights)
    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)

    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)

    x_train = np.arange(len(total_train_loss))
    x_val = np.arange(len(total_val_loss))
    
    sns.set_style('whitegrid')
    plt.figure(figsize=(14,5))
    
    plt.subplot(1,2,1)
    sns.lineplot(x=x_train, y=total_train_loss, label='Training Loss')
    sns.lineplot(x=x_val, y=total_val_loss, label='Validation Loss')
    plt.title("Loss over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(len(total_train_loss)))
    
    plt.subplot(1,2,2)
    sns.lineplot(x=x_train, y=train_accuracies, label='Training Accuracy')
    sns.lineplot(x=x_val, y=val_accuracies, label='Validation Accuracy')
    plt.title("Accuracy over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(len(total_train_loss)))

    plt.show()
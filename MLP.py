import torch
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
from keras.utils import to_categorical


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, type):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.type = type
        layers = []
        for i in range(len(self.hidden_size)):
            if i == 0:
                layers.append(nn.Linear(input_size, self.hidden_size[i]))
            else:
                layers.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self.hidden_size[-1], self.output_size)

    def forward(self, x):
        out = self.hidden_layers(x)
        if self.type == 'cls_multi':
            out = self.softmax(self.output_layer(out))
        elif self.type == 'cls_bin':
            out = self.sigmoid(self.output_layer(out))
        elif self.type == 'reg':
            out = self.output_layer(out)
        return out

    def data_system(self, batch_train, batch_test, X_train, y_train, X_test, y_test):
        # Con la seguente funzione, basta inserire al suo interno il dataset splittato in train e test (anche se sono numpy array), ricevendo in output i dati divisi in batch come tensori torch, in aggiunta con il dataloader per il train e test set
        if self.type == 'cls_multi' or self.type == 'cls_bin':
            y_train = to_categorical(y_train, np.max(y_train) + 1)
            y_test = to_categorical(y_test, np.max(y_test) + 1)
        X_train = torch.Tensor(X_train).float()
        X_test = torch.Tensor(X_test).float()
        y_train = torch.Tensor(y_train).float()
        y_test = torch.Tensor(y_test).float()
        train_tensor = TensorDataset(X_train, y_train)  # a datset are pair of feature tensor, target tensor
        dataloader_train = DataLoader(train_tensor, batch_size=batch_train, shuffle=True, num_workers=1, drop_last=True)
        test_tensor = TensorDataset(X_test, y_test)  # a datset are pair of feature tensor, target tensor
        dataloader_test = DataLoader(test_tensor, batch_size=batch_test, shuffle=True, num_workers=1, drop_last=True)
        X_train, y_train = next(iter(dataloader_train))
        X_test, y_test = next(iter(dataloader_test))
        return X_train, y_train, X_test, y_test, dataloader_train, dataloader_test

    def compile(self, modello, X):
        global criterion
        global optimizer
        if self.type == 'cls_multi' or self.type == 'cls_bin':
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(modello.parameters(), lr=0.001)
        elif self.type == 'reg':
            criterion = nn.MSELoss()
            optimizer = optim.Adam(modello.parameters(), lr=0.001)
        if torch.cuda.is_available():
            summary(modello.cuda, input_size=(X.shape[0], X.shape[1]))
        else:
            summary(modello, input_size=(X.shape[0], X.shape[1]))

    def accuracy(self, y_pred, y_true):
        _, predicted = torch.max(y_pred, 1)
        y_true = torch.argmax(y_true, dim=1)
        correct = (predicted == y_true).sum().item()
        total = y_true.size(0)
        return correct / total

    def plot(self, accuracy, loss, epoche):
        if self.type == 'cls_multi' or self.type == 'cls_bin':
            plt.subplot(2,1,1)
            plt.plot(epoche,loss, '.r', label='Train_loss in funzione delle epoche')
            plt.xlabel('epoche')
            plt.ylabel('loss')
            plt.grid(True)
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(epoche, accuracy, '.r', label='Train_accuracy in funzione delle epoche')
            plt.xlabel('epoche')
            plt.ylabel('accuracy')
            plt.grid(True)
            plt.legend()
        elif self.type == 'reg':
            plt.plot(epoche,loss, '.r', label='Train_loss in funzione delle epoche')
            plt.xlabel('epoche')
            plt.ylabel('loss')
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        plt.show()


def training(modello, X_train, y_train, epochs, dataloader, type):
    train_loss = np.zeros(epochs)
    mean_accuracy = np.zeros(epochs)
    epoche = np.zeros(epochs)
    for epoch in range(epochs):
        # Forward pass
        epoche[epoch] = epoch
        modello.train()
        batch_loss = 0.0
        total_accuracy = 0.0
        counter = 0
        for X_train, y_train in dataloader:
            counter += 1
            outputs = modello(X_train)
            batch_loss = criterion(outputs, y_train)
            if type == 'cls_multi' or type == 'cls_bin':
                total_accuracy += modello.accuracy(outputs, y_train)
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss[epoch] = batch_loss / counter
        if type == 'cls_multi' or type == 'cls_bin':
            mean_accuracy[epoch] = total_accuracy / len(dataloader)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss[epoch]:.4f}, Accuratezza: {mean_accuracy[epoch]:.4f}')
        elif type == 'reg':
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss[epoch]:.4f}')

    if type == 'cls_multi' or type == 'cls_bin': return mean_accuracy, train_loss, epoche
    elif type == 'reg': return train_loss, epoche




#%%

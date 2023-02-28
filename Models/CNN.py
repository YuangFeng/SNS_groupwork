import argparse
from tqdm import tqdm
import pandas as pd  
import numpy as np

import torch
import torch.nn as nn
from  torch.utils.data import Dataset, DataLoader
from sklearn import metrics

class ConvNet(nn.Module):
    def __init__(self, out_channels, out_dim = 3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2), #25
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2), #12
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2), #6
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        self.linear = nn.Linear(out_channels*6,out_dim)
        self.sigmod = nn.Sigmoid()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.sigmod(self.linear(x.flatten(1)))
        return x
    
class MyDataset(Dataset):
    def __init__(self, data):
        all_data = data.values
        self.data = torch.from_numpy(all_data[:,:-2]).to(torch.float32)
        self.labels = torch.from_numpy(all_data[:,-1]).long()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
    
def prepare_data(data_path, batch_size):
    name2id = dict(A=0,H=1,D=2)
    df = pd.read_csv(data_path)
    df = df.dropna()
    labels = df['Result']
    labels = labels.apply(lambda x: name2id[x])
    df = df.drop(['Data/Matchweek', 'H', 'A', 'Result', 'H points', 'A points',
       'H Goals', 'A Goals', 'H Shots', 'A Shots', 'H shots on target',
       'A shots on target', 'H fouls', 'A fouls', 'H yellows', 'A yellows',
       'H corners', 'A Corners',
       'H_avg_yellow_card5', 'A_avg_yellow_card5','H_avg_yellow_card20','A_avg_yellow_card20'], axis=1)
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df['labels'] = labels
    # print(df)
    # print(labels)
    train_df = df.sample(frac=0.7)
    val_df =df[~df.index.isin(train_df.index)]
    train_loader = DataLoader(MyDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MyDataset(val_df), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_epoch(model, data_loader,criterion, optimizer):
    model.train()
    for x,y in tqdm(data_loader):
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def val(model, data_loader):
    pred_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for id,(x,y) in enumerate(data_loader):
            out = model(x)
            pred = out.argmax(dim=1).squeeze().cpu().detach().numpy()
            pred_list.extend(pred)
            label_list.extend(y.squeeze().cpu().detach().numpy())
    acc = metrics.accuracy_score(pred_list, label_list)
    return acc

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')
    parser.add_argument('--batch_size', default=6)
    parser.add_argument('--epoches', default=40)
    parser.add_argument('--lr', default=0.0004)

    args = parser.parse_args()
    train_loader, val_loader = prepare_data(args.data, args.batch_size)
    print(len(train_loader.dataset), len(val_loader.dataset))
    model = ConvNet(8)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for i in range(args.epoches):
        train_epoch(model, train_loader, criterion, optimizer)
        acc = val(model, val_loader)
        if acc > best_acc:
            best_acc = acc #best acc， acc of 55.46%
            torch.save(model, 'Best_CNN.pth')#save the model has best performance
            print('model saved!')
        print('epoch [{}/{}] acc:{} best_acc:{}'.format(i+1, args.epoches, acc, best_acc))

if __name__ == '__main__':
    train()
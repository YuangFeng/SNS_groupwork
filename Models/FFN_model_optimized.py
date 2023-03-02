import argparse
from tqdm import tqdm
import pandas as pd  
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

class FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers) -> None:
        super().__init__()
        self.ffn =  nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        for i in range(num_layers-2):
            self.ffn.append(nn.Linear(hidden_dim, hidden_dim),)
            self.ffn.append(nn.ReLU())#activation function
        self.ffn.append(nn.Linear(hidden_dim, out_dim))
        self.ffn.append(nn.Sigmoid())
        
    def forward(self, x):
        out = self.ffn(x)
        return out

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
    name2id = dict(A=0,H=1,D=2)#win draw or loss
    df = pd.read_csv(data_path)
    df = df.dropna()
    labels = df['Result']
    labels = labels.apply(lambda x: name2id[x])
    df = df.drop(['Data/Matchweek', 'H', 'A', 'Result', 'H points', 'A points',
       'H Goals', 'A Goals', 'H Shots', 'A Shots', 'H shots on target',
       'A shots on target', 'H fouls', 'A fouls', 'H yellows', 'A yellows',
       'H corners', 'A Corners',
       'H_avg_yellow_card5', 'A_avg_yellow_card5','H_avg_yellow_card20','A_avg_yellow_card20'], axis=1)#drop features
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df['labels'] = labels
    # print(df)
    # print(labels)
    train_df = df.sample(frac=0.75)
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

def evaluate(model, data_loader):
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
    f1 = metrics.f1_score(label_list, pred_list, zero_division=0, average='micro')
    cm =metrics.confusion_matrix(label_list, pred_list)
    roc = metrics.roc_curve(label_list, pred_list, pos_label=1)  # fpr, tpr, thresholds
    return acc, f1, cm, roc

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Users/fengyuang/Desktop/Cam CEPS/Term1/Software Network/Assignment/SNS_groupwork/Datasets/1502b.csv')
    parser.add_argument('--batch_size', default=6)
    parser.add_argument('--epoches', default=40)
    parser.add_argument('--lr', default=0.0006)#lr: learning rate

    args = parser.parse_args()
    train_loader, val_loader = prepare_data(args.data, args.batch_size)
    print(len(train_loader.dataset), len(val_loader.dataset))
    model = FFN(48,784,3,5)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for i in range(args.epoches):
        train_epoch(model, train_loader, criterion, optimizer)
        acc, f1, cm, roc = evaluate(model, val_loader)
        if acc > best_acc:
            best_acc = acc #best acc， acc of 55.46%
            torch.save(model, 'Best_FFN1.pth')#save the model has best performance
            print('model saved!')
        print('######Testing result#######')
        print('epoch [{}/{}]'.format(i+1, args.epoches))
        print('1.test acc:{}\n2.best acc:{}\n3.test f1:{}\n4.confusiom matrix:{}\n'.format(acc, best_acc, f1, cm))

if __name__ == '__main__':
    train()
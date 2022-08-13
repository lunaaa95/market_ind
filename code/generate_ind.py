import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from code.train import train, val, pred
from code.Gru_model import Gru_model
from tqdm import tqdm, trange

SEED = 0
window_sz = 30
device = "cuda:3"

class Market_dataset():
    def __init__(self, market, date):
        self.market = market.astype(np.float32)
        self.date = date
        self.length = len(date) - window_sz
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # return date[idx: idx + window_sz], market[idx: idx + window_sz]
        return self.market[idx: idx + window_sz], self.market[idx + window_sz].reshape(1, -1), self.date[idx + window_sz - 1]

def prepare_data(part_n_id):
    date = np.load('../data/raw_data/stock_date.npy', allow_pickle=True)
    date = [i for i in set(date.squeeze())]
    date.sort()
    date = np.array(date)

    market = np.load('../data/market/market_2side.npy', allow_pickle=True)

    train_dataset = Market_dataset(market[:1020, part_n_id], date[:1020]) # item: ((30, 256), (256,))
    val_dataset = Market_dataset(market[1020:, part_n_id], date[1020:])
    all_dataset = Market_dataset(market[:, part_n_id], date)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True) # item: (bsz=32, seq_len=30, channel=2, dim=256)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True) 
    all_dataloader = DataLoader(dataset=all_dataset, batch_size=32, shuffle=False)
    return train_dataloader, val_dataloader, all_dataloader

def train_model(train_dataloader, val_dataloader, all_dataloader, path):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = Gru_model(input_size=256, hidden_size=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    best_loss = 1e3
    best_epoch = -1
    for epoch in range(100):
        loss = train(model, train_dataloader, optimizer, epoch, device)
        eval_loss = val(model, val_dataloader, epoch, device)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            torch.save(model.state_dict(), path)
        print('epoch {}, loss: {:.5f}'.format(epoch, eval_loss))
    print('best epoch: {}, best loss: {:.5f}'.format(best_epoch, best_loss))

def generate_date_2_market():
    part_n = 2
    ret = []
    for i in range(part_n):
        train_dataloader, val_dataloader, all_dataloader = prepare_data(i)
        # train_model(train_dataloader, val_dataloader, all_dataloader, '../data/model' + str(i) + '.pt')
        model = Gru_model(input_size=256, hidden_size=256).to(device)
        model.load_state_dict(torch.load('../data/model' + str(i) + '.pt'))
        date_2_market = pred(model, all_dataloader, device)
        print('successfully generate date_2_market_{} dict'.format(i))
        np.save('../data/market/date_2_market_' + str(i) + '.npy', date_2_market)
        print('dict saved.')
        ret.append(date_2_market)
    return ret

if __name__ == '__main__':
    dic1, dic2 = generate_date_2_market()
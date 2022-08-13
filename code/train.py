import ipdb
import torch
import torch.nn as nn

def train(model, train_dataloader,optimizer, epoch, device):
    model.train()
    loss = 0.0
    batch_num = len(train_dataloader)
    criterion = nn.CosineEmbeddingLoss()
    for batch in train_dataloader:
        optimizer.zero_grad()
        x = batch[0].to(device)
        y = batch[1].to(device)
        pred = model(x)
        batch_loss = criterion(pred.squeeze(0), y.squeeze(1), torch.ones(y.shape[0]).to(device))
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
    return loss/batch_num

def val(model, eval_dataloader, epoch, device):
    model.eval()
    loss = 0.0
    batch_num = len(eval_dataloader)
    criterion = nn.CosineEmbeddingLoss()
    for batch in eval_dataloader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        pred = model(x)
        batch_loss = criterion(pred.squeeze(0), y.squeeze(1), torch.ones(y.shape[0]).to(device))
        loss += batch_loss.item()
    return loss/batch_num

def pred(model, all_dataloader, device):
    model.eval()
    batch_num = len(all_dataloader)
    date_2_market = dict()
    for batch in all_dataloader:
        x = batch[0].to(device)
        date = batch[2]
        pred = model(x).to('cpu').detach().numpy().squeeze(0)
        date_2_market.update(dict(zip(date, pred)))
    return date_2_market


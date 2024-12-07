import config
from utils.utils import multiclass_accuracy

import numpy as np
import torch
from tqdm import tqdm 


def train_fn(model, dataloader, optimizer, current_epoch):
    
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    progress_bar = tqdm(dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch+1) + '/' + str(config.EPOCHS))
    
    for t, data in enumerate(progress_bar):
        images, labels = data
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        logits, loss = model(images, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += multiclass_accuracy(logits, labels)
        
        temp = {'loss': '%6f' %float(total_loss/ (t+1)), 'acc': '%6f' %float(total_acc/ (t+1))}
        progress_bar.set_postfix(temp)
        
    return total_loss/len(dataloader), float(total_acc/len(dataloader))


def eval_fn(model, dataloader, current_epoch):
    
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    progress_bar = tqdm(dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch+1) + '/' + str(config.EPOCHS))
    
    with torch.no_grad():
        for t, data in enumerate(progress_bar):
            images, labels = data
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            logits, loss = model(images, labels)

            total_loss += loss.item()
            total_acc  += multiclass_accuracy(logits, labels)

            temp = {'loss': '%6f' %float(total_loss/ (t+1)), 'acc': '%6f' %float(total_acc/ (t+1))}
            progress_bar.set_postfix(temp)
        
    return total_loss/len(dataloader), float(total_acc/len(dataloader))


def fit(model, trainloader, validloader, optimizer):
    
    best_valid_loss = np.inf
    history = []
    
    for i in range(config.EPOCHS):
        train_loss, train_acc = train_fn(model, trainloader, optimizer, i)
        valid_loss, valid_acc = eval_fn(model, validloader, i)
        history.append({'epoch': i + 1, 'train_loss': train_loss, 'train_acc': train_acc,
                        'valid_loss': valid_loss, 'valid_acc': valid_acc})
        print(f"epoch {i+1}: train loss {train_loss:.4f} acc {train_acc:.4f} | valid loss {valid_loss:.4f} acc {valid_acc:.4f}")
        
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), config.WEIGHTS_PATH)
            print("SAVED BEST WEIGHTS")
            best_valid_loss = valid_loss
            
    return history

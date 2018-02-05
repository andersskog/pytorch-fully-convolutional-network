import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from CONSTANTS import NUM_EPOCHS, NUM_CLASSES, USE_CUDA
from utils import accuracy

def train_step(batch_data, model, optimizer, criterion, train=True, print_accuracy=False):
    (imgs, segs, infos) = batch_data

    # feed input data
    input_img = Variable(imgs, volatile=not is_train).cuda() if USE_CUDA else Variable(imgs, volatile=not train)
    label_seg = Variable(segs, volatile=not is_train).cuda() if USE_CUDA else Variable(segs, volatile=not train)

    # forward pass
    pred = model(input_img)
    err = criterion(pred, label_seg)

    # Backward
    if train:
        err.backward()
        optimizer.step()
    
    if print_accuracy:
        print('Accuracy: {}'.format(accuracy(batch_data, pred)))
    
    return err.data[0]
   
def train(train_loader, val_loader, model, optimizer, criterion):
    print('start training')
    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        print('epoch')
        timestamp1 = time.time()
        training_loss = 0
        val_loss = 0
        for index, batch_data in enumerate(train_loader):
            print('step')
            print_accuracy = False
            if index % 100:
                print_accuracy = True
            training_loss += train_step(batch_data, model, optimizer, criterion, train=True, print_accuracy = True)
        for index, batch_data in enumerate(val_loader):
            val_loss += train_step(batch_data, model, optimizer, criterion, train=False)
        timestamp2 = time.time()
        train_losses.append(training_loss)
        val_losses.append(val_loss)
        print('\nEpoch: {} | TRAINING Loss: {} | TESTING Loss: {} | Time: {}\n'.format(
            epoch_num + 1, training_loss, val_loss, timestamp2 - timestamp1))
        return train_losses, val_losses
import tqdm
import torch
import numpy as np
from data import returnDatasAsTensor

# To get classes and stuffs
cifar10 = returnDatasAsTensor()


def one_hot_encoder(one_d_array, class_numb):
    
    empty = torch.zeros((one_d_array.shape[0], class_numb))

    for i in range(one_d_array.shape[0]):
        idx = one_d_array[i].item()
        for k in range(class_numb):
            empty[i, k][k == idx] = 1
    
    return empty

def prediction(x):
        x = torch.argmax(x, 1)
        return x

def accuracy(predict, target):
    total = 0
    true = 0
    for i in range(predict.shape[0]):
        tahmin = np.array(cifar10.classes)[predict[i].cpu()]
        hedef = np.array(cifar10.classes)[target[i].cpu()]
        if tahmin == hedef:
            true += 1
        total += 1
    return true, total



def train(model, optimizer, criterion, epochs, train_loader, test_loader, scheduler, cuda):
    _model = model
    best_val = 100.0
    validation_loss_graph = np.empty((epochs))
    train_loss_graph = np.empty((epochs))
    for epoch in range(epochs):
        running_loss = 0
        dummy = 0
        accuracy_train = 0
        for train in tqdm.tqdm(train_loader):

            batch_train_images, batch_train_labels = train
            batch_train_labels_ohe = one_hot_encoder(batch_train_labels, 10)

            if cuda:
                batch_train_labels = batch_train_labels_ohe.cuda()
                batch_train_labels = batch_train_labels.cuda()
                batch_train_images = batch_train_images.cuda()
            
            optimizer.zero_grad()
            # Prediction
            output = _model(batch_train_images)
            
            output_predict = prediction(output)
            batch_train_labels_predict = prediction(batch_train_labels)

            true_train, total_train = accuracy(output_predict, batch_train_labels_predict)
            accuracy_train += true_train / total_train * 100
            
            #Loss
            
            loss = criterion(output, batch_train_labels)
            loss.backward()

            #Optimizer
            optimizer.step()

            running_loss += loss.float()
            
            if dummy == len(train_loader) - 1:
                accuracy_val = 0
                validation_loss = 0
                for test in test_loader:

                    batch_test_images, batch_test_labels = test
                    batch_test_labels_ohe = one_hot_encoder(batch_test_labels, 10)
                    if cuda:
                        batch_test_labels_ohe = batch_test_labels_ohe.cuda()
                        batch_test_labels = batch_test_labels.cuda()
                        batch_test_images = batch_test_images.cuda()

                    with torch.no_grad():
                        outputs = _model(batch_test_images)
                        val_loss = criterion(outputs, batch_test_labels_ohe)
                    
                    outputs_predict = prediction(outputs)
                    batch_test_labels_predict = prediction(batch_train_labels)

                    true_val, total_val = accuracy(outputs_predict, batch_test_labels_predict)
                    accuracy_val += true_val / total_val * 100
                    validation_loss += val_loss.float()
            dummy += 1

        validation_loss_graph[epoch] = (validation_loss / len(test_loader))
        train_loss_graph[epoch] = (running_loss / len(train_loader))
        print(f'\nEpoch {epoch}')
        print(f'Running loss is {running_loss},\n\nAccuracy of train is % {accuracy_train / len(train_loader)}, \nAccuracy of test is % {accuracy_val / len(test_loader)}\n')
        print(f'Training loss is {train_loss_graph[epoch]}, Validation loss is {validation_loss_graph[epoch]}')
        if val_loss < best_val:
            print(f'Best model will be saved. Epoch is {epoch}.')
            best_model = _model
            best_val = val_loss
        scheduler.step(validation_loss_graph[epoch])

    return best_model, best_val, train_loss_graph, validation_loss_graph[epoch]


            


from data import getDataLoader
from model_vgg import VGGModel
from train import train
from torch import optim
import torch.nn as nn

if __name__ == '__main__':
    #################

    epoch = 1
    batch_size = 8

    #################

    train_loader, test_loader = getDataLoader(batch_size)

    model = VGGModel(channels = 3, height = 32, weight = 32, classes = 10)

    optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)

    criterion = nn.BCELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience = 2, verbose = True)
    if True:
        model = model.cuda()
        criterion = criterion.cuda()
    train(model, optimizer, criterion, epoch, train_loader, test_loader,scheduler, cuda = True)

    dummy = torch.tensor([1, 0, 2, 3, 7, 9, 6])
    print('data..: ', dummy)
    print('data.size..: ', dummy.shape)

    dummy = one_hot_encoder(dummy, 10)
    print('data...: ', dummy)
    print('doata.size...: ', dummy.shape)
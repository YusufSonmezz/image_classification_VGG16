from data import getDataLoader
from model_vgg import VGGModel
from train import train
from torch import optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

#################

epochs = 16
batch_size = 16
cuda = True

#################

train_loader, test_loader = getDataLoader(batch_size)

model = VGGModel(channels = 3, height = 32, weight = 32, classes = 10)

optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience = 2, verbose = True)

criterion = nn.BCELoss()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

best_model, best_val, train_values, validation_values = train(model, optimizer, criterion, epochs, train_loader, test_loader, scheduler, cuda)

torch.save(best_model, r'C:\Users\Yusuf\Desktop\Deneme Klasörleri\PyTorch\Models\best_model.pth')

plt.plot(range(epochs), train_values, color = 'red', label = 'Train')
plt.plot(range(epochs), validation_values, color = 'blue', label = 'Vall')
plt.title("Loss Graph")
plt.legend()
plt.show()

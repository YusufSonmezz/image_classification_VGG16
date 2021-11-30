import data
import torch 
import numpy as np
import matplotlib.pyplot as plt

cuda = True

def prediction(x):
        x = torch.argmax(x, 1)
        return x

train_loader, test_loader = data.getDataLoader(20)
cifar10 = data.returnDatasAsTensor()

model = torch.load(r'C:\Users\Yusuf\Desktop\Deneme Klasörleri\PyTorch\Models\best_model.pth')
model.eval()

for test in test_loader:
    images, labels = test
    print(images.shape)
    images_np = np.array(images.permute(0, 2, 3, 1))
    images_np = np.squeeze(images_np, axis = 2)
    plt.imshow(images_np)
    plt.show()
    if cuda:
        images = images.cuda()
        labels = labels.cuda()
    break
out = model(images)
out = prediction(out)


total = 0
true = 0
for i in range(20):
    tahmin = np.array(cifar10.classes)[out[i].cpu()]
    hedef = np.array(cifar10.classes)[labels[i].cpu()]
    print('Tahmin ve Hedef..: ', tahmin, ' ', hedef)
    if tahmin == hedef:
        true += 1
    total += 1

accuracy = true / total * 100
print(f'Accuracy is % {accuracy}')

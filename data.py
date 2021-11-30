import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
from torch.utils.data import DataLoader
import tqdm
import torch

def returnDatasAsTensor():
    
    # Download and load the images from the CIFAR10 dataset
    cifar10_data = datasets.CIFAR10(
        root="data",            # path where the images will be stored
        download=True,          # all images should be downloaded
        transform=ToTensor()    # transform the images to tensors
        )
    return cifar10_data

def getDataLoader(batch_size):

    # Splitting training and testing data
    training_data = datasets.CIFAR10(
        root = 'data',
        train = True,
        download = True,
        transform = ToTensor(),
    )
    
    testing_data = datasets.CIFAR10(
        root = 'data',
        train = True,
        download = True,
        transform = ToTensor(),
    )

    # Creating DataLoaders
    train_loader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(testing_data, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader




if __name__ == '__main__':
    cifar10 = returnDatasAsTensor()
    print('len......:', len(cifar10))
    print('classes..:', cifar10.classes)
    image, label= cifar10[random.randint(0, len(cifar10))]
    print('type.....:', type(image))
    print('size.....:', image.shape)
    print('label....:', cifar10.classes[label])

    # Show image
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    train, test = getDataLoader()
    for trainer in tqdm.tqdm(train):
        image, label = trainer
        #print('len of images..: ', len(image))
        image = image[0, :, :, :]
        #plt.imshow(image.permute(1, 2, 0))
        #plt.show()

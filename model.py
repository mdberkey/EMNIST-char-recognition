import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import pickle
import requests
import zipfile
import os, shutil


def main(setup=True, train_model=False):
    matplotlib.rcParams['figure.facecolor'] = '#ffffff'
    project_name = 'emnist-char-recognition'

    if setup:
        dataset, test_dataset = download_dataset()
        file_id = '1PaJQxzROPvrMC-ku3hYl1Et8m_KiN2q0'
        destination = 'emnist_data.zip'
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile("emnist_data.zip", "r") as zip_ref:
            zip_ref.extractall(".")

        os.remove('emnist_data.zip')

    # show_example(dataset[0])

    random_seed = 50
    torch.manual_seed(random_seed)

    val_size = 50000
    train_size = len(dataset) - val_size

    # Dividing the dataset into training dataset and validation dataset
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Creating the training dataloader and validation dataloader wirh 400
    # batch size

    batch_size = 400

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4,
                          pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

    #show_batch(train_dl)

    device = get_default_device()
    print(device)

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    # Creating the model which take 1 channel input and return 62 channel output and loading into working runtime type

    model = to_device(ResNet9(1, 62), device)

    epochs = 8
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    history = []

    if train_model:
        print(
            "Training the model from the begning. Expected time for completion for 8 epochs is 1 hour\n\n")
        history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                                 grad_clip=grad_clip,
                                 weight_decay=weight_decay,
                                 opt_func=opt_func)
    else:
        print("Using trained pratameters\n")
        print(model.load_state_dict(torch.load('emnist-resnet9.pth',
                                               map_location=get_default_device())))
        print()

    # The the evaluation of the model over validation dataset
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size=400),
                                   device)
    result = [evaluate(model, val_dl)]
    print("The final Accuracy of model on Test Dataset:", result[0]["val_acc"])
    print("The final Loss of model on Test Dataset:    ",
          result[0]["val_loss"])

    predict_image(dataset[1340])





def download_dataset():
    """
    Downloads EMNIST dataset to data/
    :return: train and test datasets
    """
    dataset = EMNIST(root="data/", split="byclass", download=True, train=True,
                     transform=tt.Compose([
                         lambda img: tt.functional.rotate(img, -90),
                         lambda img: tt.functional.hflip(img),
                         tt.ToTensor()
                     ]))

    test_dataset = EMNIST(root="data/", split="byclass", download=True,
                          train=False,
                          transform=tt.Compose([
                              lambda img: tt.functional.rotate(img, -90),
                              lambda img: tt.functional.hflip(img),
                              tt.ToTensor()
                          ]))
    return dataset, test_dataset


def to_char(num):
    """
    Converts class index to char
    :param num: class index
    :return: corresponding char
    """
    if num < 10:
        return str(num)
    elif num < 36:
        return chr(num + 55)
    else:
        return chr(num + 61)


def to_index(char):
    """
    Converts char to class index
    :param char: 0-9 =, a-z, A-Z
    :return: corresponding class index
    """
    if ord(char) < 59:
        return ord(char) - 48
    elif ord(char) < 95:
        return ord(char) - 55
    else:
        return ord(char) - 61


def show_example(data):
    """
    Shows img and corresponding label
    :param data: pixel data
    :return: none
    """
    img, label = data
    print("Label: (" + to_char(label) + ")")
    plt.imshow(img[0], cmap="gray")
    plt.show()


def show_batch(dl):
    """
    Shows imgs of batch data
    :param dl: data loader
    :return: none
    """
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=20).permute(1, 2, 0))
        break


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


class CharacterClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {['val_acc']:.4f}")


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(CharacterClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(7),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
                                                epochs=epochs,
                                                steps_per_epoch=len(
                                                    train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def predict_image(data):
    print("Predected Character: "+ to_char(torch.max(model(to_device(data[0].unsqueeze(0), device)), dim=1)[1].item()))
    print("Labeled:" , to_char(data[1]))
    plt.imshow(data[0][0], cmap="gray")
    plt.show()

if __name__ == '__main__':
    main()

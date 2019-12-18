import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn import preprocessing
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from unet_model import UNet

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os


class SatelliteDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True, device="cpu"):
        self.image_paths = [os.path.join(image_paths, imPath) for imPath in os.listdir(image_paths)]
        self.target_paths = [os.path.join(target_paths, imPath) for imPath in os.listdir(target_paths)]
        self.train = train
        self.device = device

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        target_image = Image.open(self.target_paths[index])
        image = image.convert('RGB')
        target_image = target_image.convert('RGB')
        # transformations, e.g. Random Crop etc.
        # Make sure to perform the same transformations on image and target
        # Here is a small example: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7?u=ptrblck
        if self.train:
            image, target_image = self.transform(image, target_image)
        else:
            # Only resize
            resize = transforms.Resize(size=(512, 512))
            image = resize(image)
            target_image = resize(target_image)
            # Transform to tensor
            image = TF.to_tensor(image)
            target_image = TF.to_tensor(target_image)
        return image, target_image

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image, target_image):
        # Resize
        resize = transforms.Resize(size=(520, 520))
        image = resize(image)
        target_image = resize(target_image)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        target_image = TF.crop(target_image, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            target_image = TF.hflip(target_image)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            target_image = TF.vflip(target_image)

        # Transform to tensor
        image = TF.to_tensor(image)
        target_image = TF.to_tensor(target_image)
        return image, target_image


def accuracy(model, test_dl):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, save_every: int = None, device="cpu"):
    writer = SummaryWriter()
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_num = 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss, num = loss_batch(model, loss_func, xb, yb, opt)
            train_loss += loss
            total_num += num
        train_loss /= total_num

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"Epoch: {epoch:5d}, Time: {(time.time()-start_time)/60:.3f} min, Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}")
        # add to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        # writer.add_scalar('Accuracy/train', np.random.random(), epoch)
        # writer.add_scalar('Accuracy/test', np.random.random(), epoch)
        if save_every is not None:
            if epoch % save_every == 0:
                # save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, f"models/model_epoch_{epoch}.pt")


class FlatLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def get_my_model():
    # model = nn.Sequential(
    #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    #     nn.ReLU(),
    #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    #     nn.ReLU(),
    #     nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    #     nn.ReLU(),
    #     nn.AdaptiveAvgPool2d(1),
    #     FlatLayer(),
    # )
    # classes here are the rgb channels, because we just want to reconstruct the image
    model = UNet(n_channels=3, n_classes=3)
    return model


def train_model():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {dev}")

    train_sat_ds = SatelliteDataset("data/sat_dataset/train/sat", "data/sat_dataset/train/combined", train=True, device=dev)
    valid_sat_ds = SatelliteDataset("data/sat_dataset/validation/sat", "data/sat_dataset/validation/combined", train=True, device=dev)
    # test_sat_ds = SatelliteDataset("data/sat_dataset/test/sat", "data/sat_dataset/test/combined", train=False, device=dev)

    # Normalize/Scale only on train data. Use that scaler to later scale valid and test data
    # Pay attention to the range of your activation function! (Tanh --> [-1,1], Sigmoid --> [0,1])
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # x_train = scaler.fit_transform(x_train)
    # x_valid = scaler.transform(x_valid)
    # x_test = scaler.transform(x_test)

    batchsize = 6
    # Create Dataloaders for the dataset
    train_dl = DataLoader(train_sat_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=8)
    valid_dl = DataLoader(valid_sat_ds, batch_size=batchsize * 2, shuffle=False, pin_memory=True, num_workers=8)
    # test_dl = DataLoader(test_sat_ds, batch_size=batchsize, shuffle=False)

    # Define model (done in function)
    lr = 0.01
    model = get_my_model()
    # Put the model on GPU
    model.to(dev)
    # Define optimizer
    opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # Loss function
    loss_func = F.l1_loss
    # Training
    epochs = 1000
    fit(epochs, model, loss_func, opt, train_dl, valid_dl, save_every=25, device=dev)
    # Save model
    torch.save({'state_dict': model.state_dict()}, "models/unet_sat.pt")
    # Calculate accuracy
    # acc = accuracy(model, test_dl)
    # # len of dataloader depends on batchsize
    # print(f'Accuracy of the network on the {len(test_sat_ds)} test images: {acc}%')


def resume_training(path_to_checkpoint):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {dev}")

    train_sat_ds = SatelliteDataset("data/sat_dataset/train/sat", "data/sat_dataset/train/combined", train=True,
                                    device=dev)
    valid_sat_ds = SatelliteDataset("data/sat_dataset/validation/sat", "data/sat_dataset/validation/combined",
                                    train=True, device=dev)
    batchsize = 3
    train_dl = DataLoader(train_sat_ds, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=4)
    valid_dl = DataLoader(valid_sat_ds, batch_size=batchsize * 2, shuffle=False, pin_memory=True, num_workers=4)

    lr = 0.01
    total_epochs = 1000
    model = get_my_model()
    model.to(dev)
    opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    train_loss = checkpoint['train_loss']
    loss_func = F.l1_loss

    fit(total_epochs-epoch, model, loss_func, opt, train_dl, valid_dl, save_every=25, device=dev)
    torch.save({'state_dict': model.state_dict()}, "models/unet_sat.pt")


def test_model(path_to_model):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_my_model()
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    # set to inference mode
    model.eval()
    model.to(dev)
    test_sat_ds = SatelliteDataset("data/sat_dataset/test/sat", "data/sat_dataset/test/combined", train=False, device=dev)
    # test_dl = DataLoader(test_sat_ds, batch_size=1, shuffle=False)
    # may need to rescale!
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # x_train = scaler.fit_transform(x_train)
    # x_valid = scaler.transform(x_valid)
    # x_test = scaler.transform(x_test)

    # Go through some examples and predict output
    n = 10
    for i in tqdm(range(len(test_sat_ds))):
        fig, axs = plt.subplots(1, 3)
        inp, target = test_sat_ds[i]
        inp = inp.to(dev)
        target = target.to(dev)
        y_pred = model(inp.unsqueeze(dim=0))
        # Copy tensors to cpu again
        inp, target, y_pred = inp.cpu(), target.cpu(), y_pred.cpu()
        # Prepare for numpy operations
        inp = np.rollaxis(np.asarray(inp.squeeze()), 0, start=3)
        target = np.rollaxis(np.asarray(target.squeeze()), 0, start=3)
        y_pred = np.rollaxis(np.asarray(y_pred.squeeze().detach().numpy()), 0, start=3)

        # Scale back to 0-255
        inp = inp*255
        target = target*255
        y_pred = y_pred*255

        # Plot
        # plt.subplot(n, 3, i+1)
        axs[0].imshow(inp.astype("uint8"))
        axs[0].axis("off")
        axs[0].set(title=f"input")
        # plt.subplot(n, 3, i + 2)
        axs[1].imshow(target.astype("uint8"))
        axs[1].axis("off")
        axs[1].set(title=f"target")
        # plt.subplot(n, 3, i + 3)
        axs[2].imshow(y_pred.astype("uint8"))
        axs[2].axis("off")
        axs[2].set(title=f"predicted")
        plt.savefig(f"data/model_test_bigger/Testing {i}.png")
        plt.close()
    # plt.show()


if __name__ == "__main__":
    train_model()
    # load_model()
    # resume_training("models/model_epoch_25.pt")
    # test_model("models/model_epoch_25.pt")

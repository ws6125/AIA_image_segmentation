import math
import torch
import torch.nn as nn
import unet
import attunet
import dataset

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.ops import sigmoid_focal_loss

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
epochs = 300
batch_size = 8
lr = 0.0003
weight_decay = 1e-5
# momentum = 0.9
threshold = 0.95
checkpoints = [80, 100, 150, 200, 300]
fl_alpha = 0.25 # default: 0.25 (range: [0, 1])
fl_gamma = 2 # default: 2
fl_reduction = 'mean' # sum / mean / None

def train():
    # model
    # model = unet.UNet(n_channels = 1, n_classes = 1, use_attention = True)
    model = attunet.AttU_Net(n_channels = 1, n_classes = 1)
    model.to(device = device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    # optimizer = optim.RMSprop(model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
    # criterion = nn.BCEWithLogitsLoss()

    # dataset
    train_dataset = dataset.FeatureDataset(is_train = True)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    best_loss = float('inf')
    best_epoch = 0
    total_train_batchs = math.ceil(len(train_dataset) / batch_size)
    for epoch in range(epochs):
        model.train()

        # record loss and accs
        train_loss = []
        train_accs = []

        batch = 0
        for data, labels in train_loader:
            batch += 1
            if 0 == batch % 100:
                print(f"[ Train Batch | {batch:04d}/{total_train_batchs:04d} ]")

            optimizer.zero_grad()

            data = data.to(device = device, dtype = torch.float32)
            labels = labels.to(device = device, dtype = torch.float32)

            pred = model(data)
            # loss = criterion(pred, labels)
            loss = sigmoid_focal_loss(pred, labels, alpha = fl_alpha, gamma = fl_gamma, reduction = fl_reduction)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
            optimizer.step()

            # set mask
            pred[threshold <= pred] = 1
            pred[threshold > pred] = 0

            acc = (pred == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        # calculate loss and accs
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()

        # save model
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'[ Save Model at Epoch {best_epoch} with Loss: {best_loss} ]')

        if (epoch + 1) in checkpoints:
            torch.save(model.state_dict(), f'best_model-{best_epoch}.pt')
            print(f'[ Save Check Point {epoch} with Best Model {best_epoch} ]')

    print(f'[ Best Epoch: {best_epoch} with Loss: {best_loss} ]')

if '__main__' == __name__:
    train()

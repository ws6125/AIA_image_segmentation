import math
import torch
import torch.nn as nn
import unet
import attunet
import dataset

from collections import OrderedDict
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.ops import sigmoid_focal_loss

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
train_teacher = False
train_student = True
teacher_epochs = 300
student_epochs = 80
train_batch_size = 8
valid_batch_size = 4
ema_alpha = 0.9
lr = 0.0003
weight_decay = 1e-5
threshold = 0
checkpoints = [80, 100, 150, 200, 300]
fl_alpha = 0.25 # default: 0.25 (range: [0, 1])
fl_gamma = 2 # default: 2
fl_reduction = 'mean' # sum / mean / None

# model
# teacher_model = unet.UNet(n_channels = 1, n_classes = 1)
teacher_model = attunet.AttU_Net(n_channels = 1, n_classes = 1)
teacher_model.to(device = device)

if not train_teacher:
    teacher_model.load_state_dict(torch.load('best_teacher_model.pt', map_location = device))

# student_model = unet.UNet(n_channels = 1, n_classes = 1)
student_model = attunet.AttU_Net(n_channels = 1, n_classes = 1)
student_model.to(device = device)

best_model = None

def trainTeacher():
    optimizer = optim.Adam(teacher_model.parameters(), lr = lr, weight_decay = weight_decay)
    # criterion = nn.BCEWithLogitsLoss()

    # dataset
    train_dataset = dataset.FeatureDataset(only_mri = True)
    train_indices, valid_indices = train_dataset.randomSplit(0.2) # split dataset for training and validate
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, sampler = train_sampler)
    valid_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, sampler = valid_sampler)

    best_loss = float('inf')
    best_epoch = 0
    total_train_batchs = math.ceil(len(train_indices) / train_batch_size)
    total_valid_batchs = math.ceil(len(valid_indices) / train_batch_size)
    for epoch in range(teacher_epochs):
        teacher_model.train()

        # shuffle for the train set and the valid set
        if (0 < epoch) and (0 == epoch % 10):
            train_indices, valid_indices = train_dataset.randomSplit(0.2) # split dataset for training and validate
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, sampler = train_sampler)
            valid_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, sampler = valid_sampler)

        # record loss and accs
        train_loss = []
        train_accs = []

        batch = 0
        for data, labels in train_loader:
            batch += 1
            if 0 == batch % 100:
                print(f"[ Teacher: Train Batch | {batch:04d}/{total_train_batchs:04d} ]")

            optimizer.zero_grad()

            data = data.to(device = device, dtype = torch.float32)
            labels = labels.to(device = device, dtype = torch.float32)

            pred = teacher_model(data)
            # loss = criterion(pred, labels)
            loss = sigmoid_focal_loss(pred, labels, alpha = fl_alpha, gamma = fl_gamma, reduction = fl_reduction)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm = 10)
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

        print(f"[ Teacher: Train | {epoch + 1:03d}/{teacher_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        teacher_model.eval()

        # record loss and accs
        valid_loss = []
        valid_accs = []

        batch = 0
        for data, labels in valid_loader:
            batch += 1
            if 0 == batch % 100:
                print(f"[ Teacher: Valid Batch | {batch:04d}/{total_valid_batchs:04d} ]")

            data = data.to(device = device, dtype = torch.float32)
            labels = labels.to(device = device, dtype = torch.float32)

            with torch.no_grad():
                pred = teacher_model(data)

            # loss = criterion(pred, labels)
            loss = sigmoid_focal_loss(pred, labels, alpha = fl_alpha, gamma = fl_gamma, reduction = fl_reduction)

            # set mask
            pred[threshold <= pred] = 1
            pred[threshold > pred] = 0

            acc = (pred == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Teacher: Valid | {epoch + 1:03d}/{teacher_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_model = teacher_model.state_dict()
            torch.save(best_model, 'best_teacher_model.pt')
            print(f'[ Save Teacher Model at Epoch {best_epoch} with Loss: {best_loss} ]')

        if ((epoch + 1) in checkpoints) and (None != best_model):
            torch.save(best_model, f'best_teacher_model-{best_epoch}.pt')
            print(f'[ Save Check Point {epoch + 1} with Best Teacher Model {best_epoch} ]')

    print(f'[ Best Epoch: {best_epoch} with Loss: {best_loss} ]')

def trainStudent():
    optimizer = optim.Adam(student_model.parameters(), lr = lr, weight_decay = weight_decay)
    # criterion = nn.BCEWithLogitsLoss()

    # dataset
    train_dataset = dataset.FeatureDataset()
    valid_dataset = dataset.ValidDataset()

    train_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = valid_batch_size, shuffle = True)

    best_loss = float('inf')
    best_epoch = 0
    total_train_batchs = math.ceil(len(train_dataset) / train_batch_size)
    total_valid_batchs = math.ceil(len(valid_dataset) / valid_batch_size)
    for epoch in range(student_epochs):
        student_model.train()

        # record loss and accs
        train_loss = []
        train_accs = []

        batch = 0
        for data, labels in train_loader:
            batch += 1
            if 0 == batch % 100:
                print(f"[ Student: Train Batch | {batch:04d}/{total_train_batchs:04d} ]")

            optimizer.zero_grad()

            data = data.to(device = device, dtype = torch.float32)
            labels = labels.to(device = device, dtype = torch.float32)

            pred = student_model(data)
            # loss = criterion(pred, labels)
            loss = sigmoid_focal_loss(pred, labels, alpha = fl_alpha, gamma = fl_gamma, reduction = fl_reduction)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm = 10)
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

        print(f"[ Student: Train | {epoch + 1:03d}/{student_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        student_model.eval()

        # record loss and accs
        valid_loss = []
        valid_accs = []

        batch = 0
        for data, path in valid_loader:
            batch += 1
            if 0 == batch % 100:
                print(f"[ Student: Valid Batch | {batch:04d}/{total_valid_batchs:04d} ]")

            data = data.to(device = device, dtype = torch.float32)

            # with torch.no_grad():
            pred = student_model(data)
            labels = teacher_model(data)

            # set mask of labels
            labels[threshold <= labels] = 1
            labels[threshold > labels] = 0

            # loss = criterion(pred, labels)
            loss = sigmoid_focal_loss(pred, labels, alpha = fl_alpha, gamma = fl_gamma, reduction = fl_reduction)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm = 10)
            optimizer.step()

            # set mask of pred
            pred[threshold <= pred] = 1
            pred[threshold > pred] = 0

            acc = (pred == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Student: Valid | {epoch + 1:03d}/{student_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_model = student_model.state_dict()
            torch.save(best_model, 'best_model.pt')
            print(f'[ Save Student Model at Epoch {best_epoch} with Loss: {best_loss} ]')

        if ((epoch + 1) in checkpoints) and (None != best_model):
            torch.save(best_model, f'best_model-{best_epoch}.pt')
            print(f'[ Save Check Point {epoch + 1} with Best Student Model {best_epoch} ]')

        # ema
        student_model_dict = student_model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in teacher_model.state_dict().items():
            if key in student_model_dict.keys():
                # alpha * teacher + (1 - alpha) * student
                new_teacher_dict[key] = (
                    ema_alpha * value + (1 - ema_alpha) * student_model_dict[key]
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        teacher_model.load_state_dict(new_teacher_dict)

    print(f'[ Best Epoch: {best_epoch} with Loss: {best_loss} ]')

if '__main__' == __name__:
    if train_teacher:
        trainTeacher()

    if train_student:
        trainStudent()
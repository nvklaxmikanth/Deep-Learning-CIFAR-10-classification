import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from time import time
import argparse



# -------------------------------
# ResNet Model Definition
# -------------------------------

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.dropout(out, 0.3)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k = 2, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.C1 = 4

        self.conv1 = nn.Conv2d(3, 2 ** self.C1, kernel_size=3, stride=1,
                            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 ** self.C1)
        self.layer1 = self._make_layer(block, k * 2 ** self.C1, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, k * 2 ** (self.C1 + 1), num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, k * 2 ** (self.C1 + 2), num_blocks[2], stride = 2)
        self.linear = nn.Linear(k * 2 ** (self.C1 + 2), num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
# -------------------------------
# Dataset Definition
# -------------------------------
class GetDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        if train:
            for i in range(1, 6):
                batch_file = os.path.join(data_dir, f"data_batch_{i}")
                with open(batch_file, 'rb') as fo:
                    batch = pickle.load(fo, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.labels += batch[b'labels']
            self.data = np.concatenate(self.data, axis=0)
        else:
            batch_file = os.path.join(data_dir, "test_batch")
            with open(batch_file, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                self.data = batch[b'data']
                self.labels = batch.get(b'labels', None)
        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform:
            img = self.transform(img)
        label = self.labels[index] if self.labels is not None else -1
        return img, label

# -------------------------------
# Model Setup
# -------------------------------
def block_count(depth: int) -> int:
    assert (depth - 4) % 6 == 0
    return (depth - 4) // 6

def get_num_blocks(depth: int) -> list:
    return [block_count(depth), block_count(depth), block_count(depth)]

def make_model(k=2, d=82):
    model = ResNet(BasicBlock, get_num_blocks(d), k=k)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Deep Learning Project-1")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='no. of epochs')
    parser.add_argument('-w','--num_workers',type=int,default=12,help='number of workers')
    parser.add_argument('-b','--batch_size',type=int,default=128,help='batch_size')
    args = parser.parse_args()   

    # hyperparams
    num_workers = args.num_workers
    batch_size = args.batch_size
    n_epochs = args.epochs

    # define transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    

    data_dir = "./dataset/cifar-10-python/cifar-10-batches-py"

    train_dataset = GetDataset(data_dir=data_dir, train=True, transform=transform_train)
    test_dataset = GetDataset(data_dir=data_dir, train=False, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
   
    model = make_model()
    summary(model, (3, 32, 32))

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs)

    # define training loop
    test_loss_min = np.Inf

    train_loss_list = list()
    test_loss_list = list()
    train_acc_list = list()
    test_acc_list = list()

    start = time()

    checkpoint_dir = "./checkpoints"
    best_model_dir = "./best_model"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    best_acc = 0  # Track best accuracy
    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}/{}'.format(epoch, n_epochs))
        train_loss = 0
        test_loss = 0
        total_correct_train = 0
        total_correct_test = 0
        total_train = 0
        total_test = 0
        # train model
        model.train()
        for data, target in train_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            # calculate accuracies
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
            total_correct_train += np.sum(correct)
            total_train += correct.shape[0]

        # validate model
        model.eval()
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                # calculate accuracies
                _, pred = torch.max(output, 1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
                total_correct_test += np.sum(correct)
                total_test += correct.shape[0]

        # update scheduler
        scheduler.step()

        # compute average loss
        train_loss /= total_train
        test_loss /= total_test

        # compute accuracies
        train_acc = total_correct_train / total_train * 100
        test_acc = total_correct_test / total_test * 100

        # save data
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # display stats
        print('Epoch: {}/{} \tTrain Loss: {:.6f} \tTest Loss: {:.6f} \tTrain Acc: {:.2f}% \tTest Acc: {:.2f}%'.format(epoch, n_epochs, train_loss, test_loss, train_acc, test_acc))

        # save best model if loss is minimised
        if test_loss <= test_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}. Saving model...'.format(test_loss_min, test_loss))

            if not os.path.isdir('best_model'):
                os.mkdir('best_model')

            torch.save(model.state_dict(), './best_model/model_with_min_loss.pt')
            test_loss_min = test_loss

        # Save checkpoint for the current epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_acc
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

        # Save the best model if accuracy improves
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(best_model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}% at {best_model_path}")


    end = time()

    print('Time elapsed: {} hours'.format((end - start) / 3600.0))

    model = make_model()

    # test model
    test_loss = 0
    total_correct = 0
    total = 0

    model.eval()
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            # calculate accuracies
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
            total_correct += np.sum(correct)
            total += correct.shape[0]

    # calculate overall accuracy
    print('Model accuracy on test dataset: {:.2f}%'.format(total_correct / total * 100))


    if not os.path.isdir('results'):
        os.mkdir('results')


    # plot and save figures
    plt.figure()
    plt.plot(np.arange(n_epochs), train_loss_list)
    plt.plot(np.arange(n_epochs), test_loss_list)
    plt.title('Learning Curve: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.savefig('./results/loss.png')
    plt.close()

    plt.figure()
    plt.plot(np.arange(n_epochs), train_acc_list)
    plt.plot(np.arange(n_epochs), test_acc_list)
    plt.title('Learning Curve: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.savefig('./results/accuracy.png')
    plt.close()


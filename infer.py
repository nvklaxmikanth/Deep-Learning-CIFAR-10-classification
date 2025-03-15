import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import pickle


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
        

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def block_count(depth: int) -> int:
    assert (depth - 4) % 6 == 0
    return (depth - 4) // 6

def get_num_blocks(depth: int) -> list:
    return [block_count(depth), block_count(depth), block_count(depth)]

def make_model(k = 2, d = 82):
    # instantiate model
    model = ResNet(BasicBlock, get_num_blocks(d), k = k)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print('cuda')
        if torch.cuda.device_count() > 1:
            print('cuda: {}'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model.to(device)
    return model

def main():
    model_path = "./best_model/best_model.pth"#trained model path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test batch
    cifar10_batch = load_cifar_batch('./dataset/cifar_test_nolabel.pkl')
    images = cifar10_batch[b'data']

    # Define the test transform
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Initialize model
    model = make_model()
    model.load_state_dict(torch.load(model_path))

    model.eval()

    # Perform inference
    ids = []
    labels = []
    with torch.no_grad():
        for idx, img in enumerate(images):
            img = transform_test(img).unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
            ids.append(idx)
            labels.append(pred)

    # Save predictions to CSV
    df = pd.DataFrame({'ID': ids, 'Labels': labels})
    df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()
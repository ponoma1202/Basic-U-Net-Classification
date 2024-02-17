import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Model 

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):       # paper uses bilinear interpolation during upsampling to make up for the loss of pixel information
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, 64, 3),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.ReLU())
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.double_conv(x)     # no padding: x1 = (2, 64, 28, 28)
        x2 = self.down1(x1)          # x2 = (2, 128, 12, 12)
        x3 = self.down2(x2)          # x3 = (2, 256, 4, 4)
        x4 = self.down3(x3)          # x4 = (2, 512, ) don't have enough to downsample
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)         
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2)    
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)     # 3 * 3 convolutions reduce width and height too much for small image
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)    # add padding to minimize reduction
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Copied the concatenation part from original code. Don't want to deal with the math and padding issues.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

batch_size = 2
lr = 0.01
epochs = 10
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = UNet(in_channels=3, out_channels=23)           # 20 classes in PASCAL VOC 2012
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training

model.train()
for epoch in range(epochs):
    accuracy = 0
    for batch_num, pairs in enumerate(trainloader):
        img, label = pairs
        optimizer.zero_grad()       # Get rid of residual gradients
        output = model(img)

        loss = criterion(output, label)     # image segmentation. not classification
        loss.backward()
        optimizer.step()

        pred = torch.argmax(dim=1)
        accuracy += np.count_nonzero(pred == label)
    total_accuracy = accuracy / len(trainset)
    print("Accuracy at epoch ", epoch, " is ", total_accuracy)

# Evaluation
model.eval()
with torch.no_grad():
    accuracy = 0
    for batch_num, pairs in enumerate(testloader):
        img, label = pairs
        output = model(img)
        pred = torch.argmax(dim=1)

        accuracy += np.count_nonzero(pred == label)
    total_accuracy = accuracy / len(testset)
        
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

        # added layers to reduce size of image to 1x1 for classification
        self.reduce_size = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2),    # result = (batch_size, 10, 13, 13)
                                        nn.ReLU(),
                                        nn.Conv2d(out_channels, out_channels, 3, stride=2),    # result = (batch_size, 10, 5, 5)
                                        nn.ReLU(),
                                        nn.Conv2d(out_channels, out_channels, 3, stride=2),    # result = (batch_size, 10, 2, 2)
                                        nn.ReLU(), 
                                        nn.Conv2d(out_channels, out_channels, 2))    # result = (batch_size, 10, 1, 1)

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
        x = self.conv(x)
        x = self.reduce_size(x)
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

# taken from pytorch tutorial
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)             # can't have more than 1 num_workers on local computer

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = UNet(in_channels=3, out_channels=10)       # CIFAR-10 has 10 classes    
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training

model.train()
for epoch in range(epochs):
    accuracy = 0
    for batch_num, pairs in enumerate(trainloader):
        img, label = pairs
        optimizer.zero_grad()       # Get rid of residual gradients
        logits = model(img).squeeze()       # remove extra dimensions
        logits = torch.nn.functional.softmax(logits, dim=1)     # softmax for cross entropy loss for classification
        #output = torch.argmax(logits, dim=1)
        
        # take majority vote to get label for entire image
        # pred = torch.flatten(output, start_dim=1)
        # final_pred = torch.zeros(batch_size)
        # for batch in range(batch_size):
        #     final_pred[batch] = torch.argmax(torch.bincount(pred[batch]))
        # final_pred = final_pred.float()

        loss = criterion(logits, label)     # classification not image segmentation
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=1)

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
        
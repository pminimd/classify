from mura_net import *
from data_loader import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PATH = 'mura_net.pth'

net = Mura_net()
mura_dataset = Mura_Classify_Dataset("/home/wenjun/Documents/检测网络/Mura_Dataset")
dataloader = DataLoader(mura_dataset, batch_size=10,
                        shuffle=True, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
net.train()

for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0]
        # print(inputs.shape)
        labels = data[1]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 3:.3f}')
            torch.save(net, "model_torch_dict1/"+str(epoch)+"loss"+str(running_loss)+PATH)

            running_loss = 0.0

print('Finished Training')
torch.save(net, PATH)

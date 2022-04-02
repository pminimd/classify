import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mura_net(nn.Module):

    def __init__(self):

        super().__init__()

        self.first_layer = nn.Conv2d(3,6,kernel_size=3,padding=1)
        self.prelu1 = nn.PReLU()
        self.depthwise_layer = nn.Conv2d(6,6,kernel_size=3,padding=1,groups=6)
        self.pointwise_layer = nn.Conv2d(6,2,kernel_size=1)
        self.prelu2 = nn.PReLU()
        self.middle_layer = nn.Conv2d(2,2,kernel_size=3,padding=1,stride=2)
        self.last_layer1 = nn.Conv2d(2,1,kernel_size=3,padding=1,stride=2)
        self.last_layer2 = nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2)
        self.Linear_1 = nn.Linear(4,2)
        # self.Linear_2 = nn.Linear(1,1)
    
    def forward(self, x):
        x = self.first_layer(x)
        # print(x.shape)
        x = self.prelu1(x)
        x = self.depthwise_layer(x)
        x = self.pointwise_layer(x)
        # print(x.shape)
        x = self.prelu2(x)
        x = self.middle_layer(x)
        x = self.last_layer1(x)
        x = self.last_layer2(x)
        # print(x.shape)
        x = self.Linear_1(x.view(-1, 4))
        # print(x.shape)
        # x = self.Linear_2(x)
        # print(x)
        return x

if __name__ == '__main__':
    mura_net = Mura_net()
    input_x = torch.rand(1, 3, 12, 12)
    output = mura_net.forward(input_x)
    
    print(output.shape)
    print(output)

    torch.save(mura_net, 'model.pth')
    model = torch.load('/home/wenjun/Documents/检测网络/mura_net/995loss0.0023732536064926535mura_net.pth')
    model.eval()
    model_path = "test_95.onnx"

    torch.onnx.export(model,input_x,model_path,opset_version=10,input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}})

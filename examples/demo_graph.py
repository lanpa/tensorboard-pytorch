import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


dummy_input = Variable(torch.rand(13, 1, 28, 28))

model = Net1()
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model, (dummy_input, ))

model = Net2()
with SummaryWriter(comment='Net2') as w:
    w.add_graph(model, (dummy_input, ))


dummy_input = Variable(torch.rand(1, 3, 224, 224))

with SummaryWriter(comment='alexnet') as w:
    model = torchvision.models.alexnet()
    w.add_graph(model, (dummy_input, ))

with SummaryWriter(comment='vgg19') as w:
    model = torchvision.models.vgg19()
    w.add_graph(model, (dummy_input, ))

with SummaryWriter(comment='densenet121') as w:
    model = torchvision.models.densenet121()
    w.add_graph(model, (dummy_input, ))

with SummaryWriter(comment='resnet18') as w:
    model = torchvision.models.resnet18()
    w.add_graph(model, (dummy_input, ))


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        return x * 2


model = SimpleModel()
dummy_input = (torch.zeros(1, 2, 3),)

with SummaryWriter(comment='constantModel') as w:
    w.add_graph(model, dummy_input)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


dummy_input = torch.rand(1, 3, 224, 224)

with SummaryWriter(comment='basicblock') as w:
    model = BasicBlock(3, 3)
    w.add_graph(model, (dummy_input, ))  # , verbose=True)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(
            n_categories +
            input_size +
            hidden_size,
            hidden_size)
        self.i2o = nn.Linear(
            n_categories +
            input_size +
            hidden_size,
            output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_letters = 100
n_hidden = 128
n_categories = 10
rnn = RNN(n_letters, n_hidden, n_categories)
cat = torch.Tensor(1, n_categories)
dummy_input = torch.Tensor(1, n_letters)
hidden = torch.Tensor(1, n_hidden)


out, hidden = rnn(cat, dummy_input, hidden)
with SummaryWriter(comment='RNN') as w:
    w.add_graph(rnn, (cat, dummy_input, hidden), verbose=False)


import pytest
print('expect error here:')
with pytest.raises(Exception) as e_info:
    dummy_input = torch.rand(1, 1, 224, 224)
    with SummaryWriter(comment='basicblock_error') as w:
        w.add_graph(model, (dummy_input, ))  # error

'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['ResNet18_cifar10BN','ResNet18_cifar100BN','ResNetNoBN','ResNet18_cifar100','ResNet18_cifar10','BatchNormModel','NonBatchNormModel','ResNet18_cifar10_GN','ResNet18_cifar10_LN']
class SimpleNet(nn.Module):
    def __init__(self, name=None):
        super(SimpleNet, self).__init__()
        self.name=name


    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)


    def copy_params(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def switch_grads(self, enable=True):
        for i, p in self.named_parameters():
            p.requires_grad_(enable)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out5 = out5.view(out5.size()[0], -1)

        return out5

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_embedding(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def first_activations(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

def ResNet18_cifar10BN(name=None, created_time=None, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2],name='{0}_ResNet_18'.format(name), created_time=created_time, num_classes=num_classes)

def ResNet18_cifar100BN(name=None, created_time=None, num_classes=100):
    return ResNet(BasicBlock, [2,2,2,2],name='{0}_ResNet_18'.format(name), created_time=created_time, num_classes=num_classes)

class BasicBlockNoBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetNoBN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=True)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def switch_grads(self, enable=True):
        for _, p in self.named_parameters():
            p.requires_grad_(enable)

    def features(self, x):
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_embedding(self, x):
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

    def first_activations(self, x):
        return self.relu(self.conv1(x))

def ResNet18_cifar10():
    return ResNetNoBN(BasicBlockNoBN, [2, 2, 2, 2], num_classes=10)

def ResNet18_cifar100():
    return ResNetNoBN(BasicBlockNoBN, [2,2,2,2], num_classes=100)

class BatchNormModel(nn.Module):
    def __init__(self, input_size=784, hidden_dim=256, output_size=10):

        super(BatchNormModel, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_dim * 2, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * 2)


        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class NonBatchNormModel(nn.Module):
    def __init__(self, input_size=784, hidden_dim=256, output_size=10):
        super(NonBatchNormModel, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

def ResNet18_cifar10_GN(num_classes=10, groups=8):
    class BasicBlockGN(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()

            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
            self.gn1 = nn.GroupNorm(groups, planes)

            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.gn2 = nn.GroupNorm(groups, planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.GroupNorm(groups, planes)
                )

        def forward(self, x):
            out = F.relu(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    return ResNetNoBN(BasicBlockGN, [2, 2, 2, 2], num_classes=num_classes)
def ResNet18_cifar10_LN(num_classes=10):
    class LayerNorm2d(nn.Module):
        def __init__(self, num_channels):
            super().__init__()
            self.ln = nn.LayerNorm(num_channels)

        def forward(self, x):
            x = x.permute(0, 2, 3, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)
            return x

    class BasicBlockLN(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()

            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
            self.ln1 = LayerNorm2d(planes)

            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.ln2 = LayerNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    LayerNorm2d(planes)
                )

        def forward(self, x):
            out = F.relu(self.ln1(self.conv1(x)))
            out = self.ln2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    return ResNetNoBN(BasicBlockLN, [2, 2, 2, 2], num_classes=num_classes)
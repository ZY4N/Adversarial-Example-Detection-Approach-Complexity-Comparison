import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_info import MNIST, CIFAR10
from utils import flop_estimator, estimate_params, estimate_inference_flops, estimate_training_flops

class CCGAN_flop_estimator(flop_estimator):

    def __init__(self):
        super().__init__("CCGAN")


    def mnist_params_impl(self):
        G_params = estimate_params(
            ccgan_mnist_generator(), (1, 60)
        )
        D_params = estimate_params(
            ccgan_mnist_descriminator(), (1, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )
        return G_params + D_params + 1 # weights + threshold


    def cifar10_params_impl(self):
        G_params = estimate_params(
            ccgan_cifar10_generator(), (1, 500 + CIFAR10.classes)
        )
        D_params = estimate_params(
            ccgan_cifar10_descriminator(), (1, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )
        return G_params + D_params + 1 # weights + threshold


    def mnist_train_flops_impl(self):

        samples = MNIST.training_samples
        batch_size = 256
        epochs = 60
        
        G = ccgan_mnist_generator()
        D = ccgan_mnist_descriminator()

        G_flops = estimate_inference_flops(
            G, (batch_size, 60)
        )

        D_flops = estimate_inference_flops(
            D, (batch_size, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )

        gan_training_flops = estimate_training_flops(
            G_flops + D_flops,
            batch_size,
            samples,
            epochs
        )

        return gan_training_flops


    def mnist_test_flops_impl(self):

        batch_size = 1
        generator_iterations = 300

        G = ccgan_mnist_generator()
        D = ccgan_mnist_descriminator()

        G_flops = estimate_inference_flops(
            G, (batch_size, 60)
        )

        D_flops = estimate_inference_flops(
            D, (batch_size, MNIST.img_channels, MNIST.img_width, MNIST.img_height),
        )

        G_fitting_flops = estimate_training_flops(
            G_flops,
            batch_size,
            generator_iterations,
            1
        )

        return {
            "D-AD": D_flops,
            "D-GAN": G_fitting_flops,
            "D-ALL": G_fitting_flops + D_flops
        }

    def cifar10_train_flops_impl(self):

        samples = CIFAR10.training_samples
        batch_size = 256
        epochs = 90

        compute_divisor = 1

        G = ccgan_cifar10_generator()
        D = ccgan_cifar10_descriminator()

        G_flops = estimate_inference_flops(
            G, (int(batch_size / compute_divisor), 500 + CIFAR10.classes)
        )

        D_flops = estimate_inference_flops(
            D, (int(batch_size / compute_divisor), CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )

        gan_training_flops = estimate_training_flops(
            (G_flops + D_flops) * compute_divisor,
            batch_size,
            samples,
            epochs
        )

        return gan_training_flops


    def cifar10_test_flops_impl(self):

        batch_size = 1
        generator_iterations = 400 # paper did not provide information about cifar fitting iterations, this is only a best guess

        G = ccgan_cifar10_generator()
        D = ccgan_cifar10_descriminator()

        G_flops = estimate_inference_flops(
            G, (batch_size, 500 + CIFAR10.classes)
        )

        D_flops = estimate_inference_flops(
            D, (batch_size, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )

        G_fitting_flops = estimate_training_flops(
            G_flops,
            batch_size,
            generator_iterations,
            1
        )
        
        return {
            "D-AD": D_flops,
            "D-GAN": G_fitting_flops,
            "D-ALL": G_fitting_flops + D_flops
        }


class ccgan_mnist_generator(nn.Module):
    def __init__(self, input_dim=50, class_dim=MNIST.classes, filter_nums=[384, 192, 96, 48, MNIST.img_channels], strides=[1, 2, 2, 2], kernel_size=4):
        super(ccgan_mnist_generator, self).__init__()
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.filter_nums = filter_nums
        self.strides = strides
        self.kernel_size = kernel_size
        self.linear_layer = nn.Linear(self.class_dim + self.input_dim, self.filter_nums[0])

        layers = []
        for i in range(1, len(self.filter_nums)):
            padding = 0 if self.strides[i-1] == 1 else 1
            layers += [
                nn.ConvTranspose2d(
                    self.filter_nums[i-1],
                    self.filter_nums[i],
                    self.kernel_size,
                    self.strides[i-1],
                    padding,
                    bias=False
                ),
                nn.BatchNorm2d(self.filter_nums[i]),
                nn.ReLU(True)
            ]

        layers.append(nn.Tanh())

        self.hidden_layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear_layer(x)
        x = x.view(-1, self.filter_nums[0], 1, 1)
        x = self.hidden_layer(x)
        return (x + 1) / 2


class ccgan_mnist_descriminator(nn.Module):
    def __init__(self, input_channel=MNIST.img_channels, filter_nums=[16, 32, 64, 128, 256, 512], strides=[2, 1, 2, 1, 2, 1], kernel_size=3):
        super(ccgan_mnist_descriminator, self).__init__()

        self.fc_dis = nn.Linear(4 * 4 * 512, 1)
        self.fc_aux = nn.Linear(4 * 4 * 512, MNIST.classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        layers = [
            nn.Conv2d(
                input_channel,
                filter_nums[0],
                kernel_size,
                strides[0],
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=False)
        ]
        for i in range(1, len(filter_nums)):
            layers += [
                nn.Conv2d(
                    filter_nums[i-1],
                    filter_nums[i],
                    kernel_size,
                    strides[i],
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(filter_nums[i]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5, inplace=False)
            ]

        self.hidden_layer = nn.Sequential(*layers)

    def forward(self, features, extract_feature=False):

        x = self.hidden_layer(features)
        x = x.view(-1, 4*4*512)
        dis = self.fc_dis(x)

        aux = self.fc_aux(x)
        classes = self.softmax(aux)
        real = self.sigmoid(dis).view(-1, 1).squeeze(1)
        if extract_feature:
            return real, classes, x
        return real, classes


class GBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, first=False):
        super(GBasicBlock, self).__init__()

        if first:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=1, padding=0, bias=False)
        elif stride != 1:

            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=stride-1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=1, padding=1,
                                            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            if first:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, self.expansion * planes, kernel_size=4, stride=stride-1,
                                       padding=stride -2, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=4, stride=stride,
                                   padding=stride-1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ccgan_cifar10_generator(nn.Module):
    def __init__(self, in_dim=500, block=GBasicBlock, num_blocks=(2, 2, 2, 2)):
        super(ccgan_cifar10_generator, self).__init__()
        self.in_planes = 1024
        self.linear = nn.Linear(in_dim + CIFAR10.classes, 1024)
        self.layer1 = self._make_layer(block, 512, num_blocks[0], stride=2, first=True)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.out_layer = nn.ConvTranspose2d(64, CIFAR10.img_channels, 3, 1, 1)

    def _make_layer(self, block, planes, num_blocks, stride, first=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if stride==2 and first:
                layers.append(block(self.in_planes, planes, stride, first=True))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.view([-1, 1024, 1, 1])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.tanh(self.out_layer(x))
        return (x + 1) / 2


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ccgan_cifar10_descriminator(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2]):
        super(ccgan_cifar10_descriminator, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(CIFAR10.img_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4_1 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.in_planes = 256
        self.layer4_2 = self._make_layer(block, 512, num_blocks[3], stride=2)

        #block.expansion = 64 * 64
        self.linear = nn.Linear(512*block.expansion, CIFAR10.classes)
        self.linear_re = nn.Linear(512*block.expansion, 1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out1 = self.layer4_1(out)
        out1 = F.avg_pool2d(out1, 4)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.layer4_2(out)
        out2 = F.avg_pool2d(out2, 4)
        out2 = out2.view(out2.size(0), -1)

        out_aux = self.linear(out1)
        out_dis = self.linear_re(out2)
        classes = self.softmax(out_aux)
        real = self.sigmoid(out_dis).view(-1, 1).squeeze(1)
        
        return real, classes

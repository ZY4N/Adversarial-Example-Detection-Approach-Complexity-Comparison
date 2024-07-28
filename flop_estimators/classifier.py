import torch.nn as nn

from dataset_info import MNIST, CIFAR10
from utils import flop_estimator, estimate_params, estimate_inference_flops, estimate_training_flops


class classifier_flop_estimator(flop_estimator):

    def __init__(self):
        super().__init__("classifier")


    def mnist_params_impl(self):
        return estimate_params(
            classifier_mnist(), (1, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )


    def cifar10_params_impl(self):
        return estimate_params(
            classifier_cifar10(), (1, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )


    def mnist_train_flops_impl(self):

        samples = MNIST.training_samples
        batch_size = 100
        epochs = 90

        classifier = classifier_mnist()

        inference_flops = estimate_inference_flops(
            classifier, (batch_size, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )

        training_flops = estimate_training_flops(
            inference_flops,
            batch_size,
            samples,
            epochs
        )

        return training_flops


    def mnist_test_flops_impl(self):

        batch_size = 1

        classifier = classifier_mnist()

        return estimate_inference_flops(
            classifier, (batch_size, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )

    def cifar10_train_flops_impl(self):

        samples = CIFAR10.training_samples
        batch_size = 100
        epochs = 120
    
        classifier = classifier_cifar10()

        inference_flops = estimate_inference_flops(
            classifier, (batch_size, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )

        training_flops = estimate_training_flops(
            inference_flops,
            batch_size,
            samples,
            epochs
        )

        return training_flops


    def cifar10_test_flops_impl(self):

        batch_size = 1

        classifier = classifier_cifar10()

        return estimate_inference_flops(
            classifier, (batch_size, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )


def _make_vgg_layers(in_channels, cfg, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def mnist_features(in_channels=MNIST.img_channels):
    return _make_vgg_layers(
        in_channels=in_channels,
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], # VGG13 without last max ppoling layer
        batch_norm=True
    )

def mnist_classifier():
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),

        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 10)
    )

class classifier_mnist(nn.Module):

    last_layer_size = 512

    def __init__(self):
        super(classifier_mnist, self).__init__()
        self.features = mnist_features()
        self.classifier = mnist_classifier()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def cifar10_features(in_channels=CIFAR10.img_channels):
    return _make_vgg_layers(
        in_channels=in_channels,
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # VGG19
        batch_norm=True
    )

def cifar10_classifier():
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),

        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 10)
    )

class classifier_cifar10(nn.Module):

    last_layer_size = 512

    def __init__(self):
        super(classifier_cifar10, self).__init__()
        self.features = cifar10_features()
        self.classifier = cifar10_classifier()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

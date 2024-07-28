import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

from .classifier import mnist_features, mnist_classifier, cifar10_features, cifar10_classifier, classifier_flop_estimator
from dataset_info import MNIST, CIFAR10
from utils import flop_estimator, estimate_params, estimate_inference_flops, estimate_training_flops

class SID_flop_estimator(flop_estimator):

    attacks = 3
    sample_instances = 1 + attacks

    def __init__(self):
        super().__init__("SID")


    def mnist_params_impl(self):
        dual_classifier_params = estimate_params(
            sid_wawt_classifier_mnist(), (1, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )
        detector_params = estimate_params(
            sid_detector(), (1, 2 * MNIST.classes)
        )
        return dual_classifier_params + detector_params


    def cifar10_params_impl(self):
        dual_classifier_params = estimate_params(
            sid_wawt_classifier_cifar10(), (1, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )
        detector_params = estimate_params(
            sid_detector(), (1, 2 * CIFAR10.classes)
        )
        return dual_classifier_params + detector_params


    def mnist_train_flops_impl(self):

        # matching hyper parameters of base classifier training
        samples = MNIST.training_samples
        batch_size = 100
        epochs = 90
        
        dual_classifier = sid_wawt_classifier_mnist()
        detector = sid_detector()

        dual_classifier_flops = estimate_inference_flops(
            dual_classifier, (batch_size, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )
        detector_flops = estimate_inference_flops(
            detector, (batch_size, 2 * MNIST.classes)
        )

        wawt_classifier_training_flops = estimate_training_flops(
            dual_classifier_flops,
            batch_size,
            samples,
            epochs
        )

        classifier_flops = classifier_flop_estimator().mnist_test_flops()
        detector_training_classifier_flops = samples * SID_flop_estimator.sample_instances * (
            dual_classifier_flops + classifier_flops
        )
        detector_training_flops = detector_training_classifier_flops + estimate_training_flops(
            detector_flops,
            batch_size,
            samples * SID_flop_estimator.sample_instances,
            epochs
        )

        return wawt_classifier_training_flops + detector_training_flops


    def mnist_test_flops_impl(self):

        batch_size = 1

        dual_classifier = sid_wawt_classifier_mnist()
        detector = sid_detector()

        dual_classifier_flops = estimate_inference_flops(
            dual_classifier, (batch_size, MNIST.img_channels, MNIST.img_width, MNIST.img_height)
        )
        detector_flops = estimate_inference_flops(
            detector, (batch_size, 2 * MNIST.classes)
        )

        return dual_classifier_flops + detector_flops


    def cifar10_train_flops_impl(self):

        # matching hyper parameters of base classifier training
        samples = CIFAR10.training_samples
        batch_size = 100
        epochs = 120
        
        dual_classifier = sid_wawt_classifier_cifar10()
        detector = sid_detector()

        dual_classifier_flops = estimate_inference_flops(
            dual_classifier, (batch_size, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )
        detector_flops = estimate_inference_flops(
            detector, (batch_size, 2 * CIFAR10.classes)
        )

        wawt_classifier_training_flops = estimate_training_flops(
            dual_classifier_flops,
            batch_size,
            samples,
            epochs
        )
        
        classifier_flops = classifier_flop_estimator().cifar10_test_flops()
        detector_training_classifier_flops = samples * SID_flop_estimator.sample_instances * (
            dual_classifier_flops + classifier_flops
        )
        detector_training_flops = detector_training_classifier_flops + estimate_training_flops(
            detector_flops,
            batch_size,
            samples * SID_flop_estimator.sample_instances,
            epochs
        )

        return wawt_classifier_training_flops + detector_training_flops


    def cifar10_test_flops_impl(self):

        batch_size = 1

        dual_classifier = sid_wawt_classifier_cifar10()
        detector = sid_detector()

        dual_classifier_flops = estimate_inference_flops(
            dual_classifier, (batch_size, CIFAR10.img_channels, CIFAR10.img_width, CIFAR10.img_height)
        )
        detector_flops = estimate_inference_flops(
            detector, (batch_size, 2 * CIFAR10.classes)
        )

        return dual_classifier_flops + detector_flops


def plugdata(x, Yl, Yh, mode):
    if mode == 'append':
        output = torch.zeros(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        output = output.cuda()
        output[:, 0:3, :] = Yl[:, :, :]
        output[:, 3:6, :] = Yh[0][:, 0, :, :]
        output[:, 6:9, :] = Yh[0][:, 1, :, :]
        output[:, 9:12, :] = Yh[0][:, 2, :, :]
        output = output.reshape(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
    elif mode == 'avg':
        output = torch.zeros(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        output = output.cuda()
        output[:, 0, :] = torch.mean(Yl[:, :, :], axis=1)
        output[:, 1, :] = torch.mean(Yh[0][:, 0, :, :], axis=1)
        output[:, 2, :] = torch.mean(Yh[0][:, 1, :, :], axis=1)
        output[:, 3, :] = torch.mean(Yh[0][:, 2, :, :], axis=1)
        output = output.reshape(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
    return output

def wavelets(self, x, FDmode):
    x = x.cuda().reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    Yl, Yh = self.DWT(x)
    output = plugdata(x, Yl, Yh, FDmode)
    return output

class sid_wawt_classifier_mnist(nn.Module):
    def __init__(self, wave='haar',mode='append'):
        super(sid_wawt_classifier_mnist, self).__init__()
        self.wave = wave
        self.DWT = DWTForward(J=1, wave = self.wave, mode='symmetric',Requirs_Grad=True).cuda()
        self.FDmode = mode
        self.features = mnist_features() # no multiplication as image is tiled
        self.classifier = mnist_classifier()

    def forward(self, x):
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3]) # duplicate channel so three channel wavelet transform works
        x = wavelets(self, x, self.FDmode)
        x = x[:, [0, 3, 6, 9], :, :] # drop duplicate channels
        # create tiled image from sub bands so it works with the default classifier
        tiled = torch.zeros(
            x.shape[0],
            int(x.shape[1] / 4),
            int(x.shape[2] * 2),
            int(x.shape[3] * 2)
        ).cuda()
        tiled[:, :, 0:14, 0:14] = x[:, 0:1, :, :]
        tiled[:, :, 0:14, 14:28] = x[:, 1:2, :, :]
        tiled[:, :, 14:28, 0:14] = x[:, 2:3, :, :]
        tiled[:, :, 14:28, 14:28] = x[:, 3:4, :, :]
        x = self.features(tiled)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class sid_wawt_classifier_cifar10(nn.Module):
    def __init__(self, wave='sym17',mode='append'):
        super(sid_wawt_classifier_cifar10, self).__init__()
        self.wave = wave
        self.DWT = DWTForward(J=1, wave = self.wave, mode='symmetric',Requirs_Grad=True).cuda()
        self.FDmode = mode
        self.features = cifar10_features(in_channels=CIFAR10.img_channels * 4) # multiply by wavelet sub bands
        self.classifier = cifar10_classifier()

    def forward(self, x):
        x = wavelets(self, x, self.FDmode)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class sid_detector(nn.Module):
    def __init__(self, C_Number=3, num_class=10):
        super(sid_detector, self).__init__()

        self.Relu = nn.ReLU(inplace=True)
        self.Linear1 = nn.Linear(2*num_class, 2*num_class)
        self.Linear = nn.Linear(2*num_class, C_Number)
        self.SM = nn.Softmax(dim=1)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.Linear(x)
        return x
    
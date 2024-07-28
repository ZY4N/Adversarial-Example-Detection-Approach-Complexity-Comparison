from calflops import calculate_flops

from .classifier import classifier_flop_estimator
from dataset_info import MNIST, CIFAR10
from utils import flop_estimator

class FS_flop_estimator(flop_estimator):

    def __init__(self):
        super().__init__("FS")


    def mnist_params_impl(self):
        return 1 # threshold


    def cifar10_params_impl(self):
        return 1 # threshold


    def mnist_train_flops_impl(self):
        samples = MNIST.training_samples

        inference_flops = self.mnist_test_flops()
        total_flops = samples * inference_flops

        return total_flops

    def mnist_test_flops_impl(self):

        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.mnist_test_flops()

        total_flops = (
            classifier_inference_flops + squeezer_bit_deth_flops(
                MNIST.img_width, MNIST.img_height, MNIST.img_channels
            ) +
            classifier_inference_flops + squeezer_mdeian_filter_flops(
                MNIST.img_width, MNIST.img_height, MNIST.img_channels, 2
            ) +
            classifier_inference_flops + squeezer_mdeian_filter_flops(
                MNIST.img_width, MNIST.img_height, MNIST.img_channels, 3
            )
        )

        return total_flops
        
    def cifar10_train_flops_impl(self):
        samples = CIFAR10.training_samples

        inference_flops = self.cifar10_test_flops()
        total_flops = samples * inference_flops

        return total_flops

    def cifar10_test_flops_impl(self):
        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.cifar10_test_flops()

        total_flops = (
            2 * (
                classifier_inference_flops + squeezer_bit_deth_flops(
                    CIFAR10.img_width, CIFAR10.img_height, CIFAR10.img_channels
                )
            ) +
            classifier_inference_flops + squeezer_mdeian_filter_flops(
                CIFAR10.img_width, CIFAR10.img_height, CIFAR10.img_channels, 2
            ) +
            classifier_inference_flops + squeezer_none_local_means_flops(
                CIFAR10.img_width, CIFAR10.img_height, CIFAR10.img_channels, 11, 3
            )
        )

        return total_flops
    
def squeezer_bit_deth_flops(img_width, img_height, img_channels):
    return img_width * img_height * img_channels * (
        1 + # multiplication
        1 + # rounding
        1 # division
    )

def squeezer_mdeian_filter_flops(img_width, img_height, img_channels, window_size):
    return img_width * img_height * img_channels * (
        2 * window_size # 255 values -> radix sort with O(N)
    )

def squeezer_none_local_means_flops(img_width, img_height, img_channels, window_size, patch_size):
    return img_width * img_height * img_channels * window_size**2 * patch_size**2 # based on naive implementation

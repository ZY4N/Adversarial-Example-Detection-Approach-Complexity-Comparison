from calflops import calculate_flops

from .classifier import classifier_flop_estimator, classifier_mnist, classifier_cifar10
from dataset_info import MNIST, CIFAR10
from utils import flop_estimator

class ODDS_flop_estimator(flop_estimator):

    noise_sources = 3
    noise_magnitudes = 256
    total_noises = noise_sources * noise_magnitudes
    
    def __init__(self):
        super().__init__("ODDS")


    def mnist_params_impl(self):
        # for every class ODDS stores the mean and deviation of similarity to every other class over all noise magnitudes
        return MNIST.classes * (MNIST.classes - 1) * ODDS_flop_estimator.noise_magnitudes * 2


    def cifar10_params_impl(self):
        # for every class ODDS stores the mean and deviation of similarity to every other class over all noise magnitudes
        return CIFAR10.classes * (CIFAR10.classes - 1) * ODDS_flop_estimator.noise_magnitudes * 2


    def mnist_train_flops_impl(self):

        samples = MNIST.training_samples

        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.mnist_test_flops()

        total_flops = samples * (
            classifier_inference_flops +
            _alignment_calculation_flops(
                classifier_inference_flops,
                ODDS_flop_estimator.total_noises,
                MNIST.img_width, MNIST.img_height, MNIST.img_channels, MNIST.classes,
                classifier_mnist.last_layer_size
            )
        )

        return total_flops

    def mnist_test_flops_impl(self):

        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.mnist_test_flops()

        last_layer_size_wo_pred = classifier_mnist.last_layer_size - 1

        total_flops = (
            _alignment_calculation_flops(
                classifier_inference_flops,
                ODDS_flop_estimator.total_noises,
                MNIST.img_width, MNIST.img_height, MNIST.img_channels, MNIST.classes,
                classifier_mnist.last_layer_size
            ) +
            ODDS_flop_estimator.total_noises * last_layer_size_wo_pred + # subtract mean
            ODDS_flop_estimator.total_noises * last_layer_size_wo_pred + # divide by std. deviation
            ODDS_flop_estimator.total_noises * last_layer_size_wo_pred + # mean over all noises and latent layer
            1 # threshold comparison
        )

        return total_flops
        
    def cifar10_train_flops_impl(self):

        samples = CIFAR10.training_samples

        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.cifar10_test_flops()

        total_flops = samples * (
            classifier_inference_flops +
            _alignment_calculation_flops(
                classifier_inference_flops,
                ODDS_flop_estimator.total_noises,
                CIFAR10.img_width, CIFAR10.img_height, CIFAR10.img_channels, CIFAR10.classes,
                classifier_cifar10.last_layer_size
            )
        )

        return total_flops

    def cifar10_test_flops_impl(self):

        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.cifar10_test_flops()

        last_layer_size_wo_pred = classifier_cifar10.last_layer_size - 1

        total_flops = (
            _alignment_calculation_flops(
                classifier_inference_flops,
                ODDS_flop_estimator.total_noises,
                CIFAR10.img_width, CIFAR10.img_height, CIFAR10.img_channels, CIFAR10.classes,
                classifier_cifar10.last_layer_size
            ) +
            ODDS_flop_estimator.total_noises * last_layer_size_wo_pred + # subtract mean
            ODDS_flop_estimator.total_noises * last_layer_size_wo_pred + # divide by std. deviation
            ODDS_flop_estimator.total_noises * last_layer_size_wo_pred + # mean over all noises and latent layer
            1 # threshold comparison
        )

        return total_flops
    
def _alignment_calculation_flops(
    inference_flops,
    noise_magnitudes,
    img_width,
    img_height,
    img_channels,
    classes,
    last_layer_size
):
    return (
        noise_magnitudes * (
            img_width * img_height * img_channels * 3 + # create noisy image (rnd, add, multiply)
            inference_flops + # inference on noisy image
            last_layer_size + # subtract normal prediction latent vector
            1 * classes * (2 * last_layer_size - 1) # matmul with weight matrix
        )
    )

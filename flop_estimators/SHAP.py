import torch.nn as nn

from .classifier import classifier_flop_estimator, classifier_mnist, classifier_cifar10
from dataset_info import MNIST, CIFAR10
from utils import flop_estimator, estimate_params, estimate_inference_flops, estimate_training_flops

class SHAP_flop_estimator(flop_estimator):

    # Authors did not specifiy number of background samples used.
    # Since the SHAP documentation recommends a number between 100 and 1000
    # we chose 1000 too kepp comparison fair.  
    background_samples = 1000
    attacks = 3
    
    def __init__(self):
        super().__init__("SHAP")


    def mnist_params_impl(self):
        detector_params = estimate_params(
            shap_detector(classifier_mnist.last_layer_size), (1, classifier_mnist.last_layer_size * MNIST.classes)
        )
        background_samples_params = SHAP_flop_estimator.background_samples * MNIST.img_channels * MNIST.img_width * MNIST.img_height
        return detector_params + background_samples_params


    def cifar10_params_impl(self):
        detector_params = estimate_params(
            shap_detector(classifier_cifar10.last_layer_size), (1, classifier_cifar10.last_layer_size * CIFAR10.classes)
        )
        background_samples_params = SHAP_flop_estimator.background_samples * CIFAR10.img_channels * CIFAR10.img_width * CIFAR10.img_height
        return detector_params + background_samples_params


    def mnist_train_flops_impl(self):
        samples = MNIST.training_samples * (SHAP_flop_estimator.attacks + 1) # attacks + normal example
        batch_size = 256
        epochs = 120
        
        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.mnist_test_flops()

        detector = shap_detector(classifier_mnist.last_layer_size)

        shap_value_flops = shap_deep_explainer_flops(
            MNIST.classes,
            classifier_inference_flops,
            classifier_mnist.last_layer_size,
            SHAP_flop_estimator.background_samples
        )
        
        detector_infernece_flops = estimate_inference_flops(
            detector, (batch_size, classifier_mnist.last_layer_size * MNIST.classes)
        )

        classifier_training_flops = classifier_inference_flops * samples * epochs
        shap_value_training_flops = shap_value_flops * samples * epochs
        detector_training_flops = estimate_training_flops(
            detector_infernece_flops,
            batch_size,
            samples,
            epochs
        )

        total_training_flops = (
            classifier_training_flops +
            shap_value_training_flops +
            detector_training_flops
        )

        return total_training_flops


    def mnist_test_flops_impl(self):

        batch_size = 1

        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.mnist_test_flops()

        detector = shap_detector(classifier_mnist.last_layer_size)

        shap_value_flops = shap_deep_explainer_flops(
            MNIST.classes,
            classifier_inference_flops,
            classifier_mnist.last_layer_size,
            SHAP_flop_estimator.background_samples 
        )
        
        detector_infernece_flops = estimate_inference_flops(
            detector, (batch_size, classifier_mnist.last_layer_size * MNIST.classes)
        )

        return shap_value_flops + detector_infernece_flops


    def cifar10_train_flops_impl(self):

        samples = CIFAR10.training_samples * (SHAP_flop_estimator.attacks + 1) # attacks + normal example
        batch_size = 256
        epochs = 140
        
        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.cifar10_test_flops()

        detector = shap_detector(classifier_cifar10.last_layer_size)

        shap_value_flops = shap_deep_explainer_flops(
            CIFAR10.classes,
            classifier_inference_flops,
            classifier_cifar10.last_layer_size,
            SHAP_flop_estimator.background_samples
        )
        
        detector_infernece_flops = estimate_inference_flops(
            detector, (batch_size, classifier_cifar10.last_layer_size * CIFAR10.classes),
        )

        classifier_training_flops = classifier_inference_flops * samples * epochs
        shap_value_training_flops = shap_value_flops * samples * epochs
        detector_training_flops = estimate_training_flops(
            detector_infernece_flops,
            batch_size,
            samples,
            epochs
        )

        return (
            classifier_training_flops +
            shap_value_training_flops +
            detector_training_flops
        )


    def cifar10_test_flops_impl(self):

        batch_size = 1

        classifier_estimator = classifier_flop_estimator()
        classifier_inference_flops = classifier_estimator.cifar10_test_flops()

        detector = shap_detector(classifier_cifar10.last_layer_size)

        shap_value_flops = shap_deep_explainer_flops(
            CIFAR10.classes,
            classifier_inference_flops,
            classifier_cifar10.last_layer_size,
            SHAP_flop_estimator.background_samples
        )
        
        detector_infernece_flops = estimate_inference_flops(
            detector, (batch_size, classifier_cifar10.last_layer_size * CIFAR10.classes)
        )

        return shap_value_flops + detector_infernece_flops


def shap_deep_explainer_flops(classes, inference_flops, last_layer_size, background_samples):
    return inference_flops + classes * ( # algorithm is reapeated for every class
        background_samples * (
            inference_flops + # one feed forward
            # backgpopagation for last fully-connected layer
            2 * classes + # loss function
            3 * classes + # sigmoid gradient
            2 * classes * last_layer_size + # weight gradient multiplication and addition
            2 * classes * last_layer_size # input gradient multiplication and addition
        ) +
        3 * background_samples * last_layer_size + # association calculation
        1 # normalization
    )

class shap_detector(nn.Module):
    def __init__(self, classifier_penultimate_layer_size):
        super(shap_detector, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(classifier_penultimate_layer_size * MNIST.classes, 256),
            nn.ReLU(True),

            nn.Linear(256, 128),
            nn.ReLU(True),

            nn.Linear(128, 16),
            nn.ReLU(True),

            nn.Linear(16, 1),
            nn.Softmax()
        )

    def forward(self, x):
        return self.classifier(x)

from calflops import calculate_flops
from thop import profile
import torch

class flop_estimator:
   
    _cache = {}
    
    def __init__(self, name):
        self.name = name

    def _cache_key(self, key):
        return f'{self.name}_{key}'
    
    def mnist_params(self):
        key = self._cache_key('mnist_params')
        if key not in flop_estimator._cache:
            flop_estimator._cache[key] = self.mnist_params_impl()
        return flop_estimator._cache[key]

    def cifar10_params(self):
        key = self._cache_key('cifar10_params')
        if key not in flop_estimator._cache:
            flop_estimator._cache[key] = self.cifar10_params_impl()
        return flop_estimator._cache[key]

    def mnist_train_flops(self):
        key = self._cache_key('mnist_train')
        if key not in flop_estimator._cache:
            flop_estimator._cache[key] = self.mnist_train_flops_impl()
        return flop_estimator._cache[key]

    def mnist_test_flops(self):
        key = self._cache_key('mnist_test')
        if key not in flop_estimator._cache:
            flop_estimator._cache[key] = self.mnist_test_flops_impl()
        return flop_estimator._cache[key]
    
    def cifar10_train_flops(self):
        key = self._cache_key('cifar10_train')
        if key not in flop_estimator._cache:
            flop_estimator._cache[key] = self.cifar10_train_flops_impl()
        return flop_estimator._cache[key]

    def cifar10_test_flops(self):
        key = self._cache_key('cifar10_test')
        if key not in flop_estimator._cache:
            flop_estimator._cache[key] = self.cifar10_test_flops_impl()
        return flop_estimator._cache[key]


    def mnist_params_impl(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def cifar10_params_impl(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def mnist_train_flops_impl(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def mnist_test_flops_impl(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def cifar10_train_flops_impl(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def cifar10_test_flops_impl(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

def estimate_params(model, input_shape):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    _, _, params = calculate_flops(
        model=model, 
        input_shape=input_shape,
        output_as_string=False
    )

    try:
        rnd_inputs = torch.randn(input_shape).to(device)
        _, params2 = profile(model, inputs=(rnd_inputs,))
        params = max(params, params2)
    except:
        print("Error while calculating weights using thop")

    return params

def estimate_inference_flops(model, input_shape):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    flops, _, _ = calculate_flops(
        model=model, 
        input_shape=input_shape,
        output_as_string=False
    )

    profile

    return flops

def estimate_training_flops(inference_flops, batch_size, samples, epochs):
    return (inference_flops / batch_size) * 3 * samples * epochs

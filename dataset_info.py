
class image_dataset_info():
    def __init__(self, img_width, img_height, img_channels, classes, training_samples, test_samples):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.classes = classes
        self.training_samples = training_samples
        self.test_samples = test_samples
        
MNIST = image_dataset_info(
    img_width=28,
    img_height=28,
    img_channels=1,
    classes=10,
    training_samples=60000,
    test_samples=10000
)

CIFAR10 = image_dataset_info(
    img_width=32,
    img_height=32,
    img_channels=3,
    classes=10,
    training_samples=50000,
    test_samples=10000
)

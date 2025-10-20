from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, DTD

def main():
    # Download CIFAR-10 dataset
    CIFAR10(root='./data', train=True, download=True)
    CIFAR10(root='./data', train=False, download=True)
    
    # Download CIFAR-100 dataset
    CIFAR100(root='./data', train=True, download=True)
    CIFAR100(root='./data', train=False, download=True)
    
    ## There can be an error when downloading EMNIST, be carefull

    # Download EMNIST-leters
    EMNIST(root='./data', split='letters', train=True, download=True)
    EMNIST(root='./data', split='letters', train=False, download=True)

    # Download EMNIST-digits
    EMNIST(root='./data', split='mnist', train=True, download=True)
    EMNIST(root='./data', split='mnist', train=False, download=True)

    # Download DTD (textures)
    DTD(root='./data', split='train', download=True)
    DTD(root='./data', split='test', download=True)

    # Tiny Imagenet must be downloaded manually from:
    # http://cs231n.stanford.edu/tiny-imagenet-200.zip
    # and unzipped in the ./data folder

if __name__ == '__main__':
    main()
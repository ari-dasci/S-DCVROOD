from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, Places365, SVHN, ImageFolder
from torch.utils.data import ConcatDataset
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class EnsureRGB:
    def __call__(self, img):
        """
        If the image has 1 channel, repeat it across 3 channels.
        """
        if img.mode == 'L':  # 'L' mode means grayscale
            img = img.convert('RGB')
        return img

def get_classes(dataset):
    # Check if the dataset has the 'classes' attribute
    if hasattr(dataset, 'classes'):
        return dataset.classes
    
    # If it doesn't have 'classes', infer from the 'targets' or 'labels' attribute
    elif hasattr(dataset, 'targets'):
        # Assuming the dataset stores target labels in a tensor/list
        targets = dataset.targets
        return sorted(set(targets.numpy())) if hasattr(targets, 'numpy') else sorted(set(targets))
    
    elif hasattr(dataset, 'labels'):
        # Some datasets (like VOC) have 'labels' instead of 'targets'
        labels = dataset.labels
        return sorted(set(labels))
    
    else:
        raise ValueError("The dataset does not have a 'classes', 'targets', or 'labels' attribute.")


class CombinedDataset(Dataset):
    def __init__(self, datasets, transforms=None):
        """
        Args:
            datasets (list of torchvision datasets): A list of torchvision datasets to be merged.
        """
        self.datasets = datasets
        self.transforms = transforms if transforms is not None else [None] * len(datasets)
        
        # Combine the classes and update label indices for each dataset
        self.classes = []
        self.dataset_ranges = []
        self.dataset_idxs = []
        offset = 0
        
        # To hold the mapping between old class indices and new class indices
        self.label_mapping = []
        #self.labels = []
        
        for didx, dataset in tqdm(enumerate(datasets), desc="Combining datasets"):
            dataset_classes = get_classes(dataset)
            # Append the current dataset's classes to the global class list
            self.classes.extend(dataset_classes)
            self.dataset_idxs.extend([didx] * len(dataset))
            
            # Save the range of indices from this dataset
            start_idx = offset
            end_idx = offset + len(dataset)
            self.dataset_ranges.append((start_idx, end_idx))
            offset += len(dataset)
            
            # Create a mapping for the class labels
            label_map = {i: i + len(self.classes) - len(dataset_classes) for i in range(len(dataset_classes))}
            self.label_mapping.append(label_map)
            # self.labels.extend([label_map[label] for label in range(len(dataset))])

            tqdm.write(f"Added {len(dataset)} samples and {len(dataset_classes)} classes from {type(dataset).__name__}")
        
        # Calculate total length
        self.total_length = sum(len(d) for d in datasets)
        print(f"Combined dataset has {len(self.classes)} classes and {self.total_length} samples.")

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """
        Given an index, return the appropriate sample and the adjusted label.
        """
        # Find out which dataset the index belongs to
        for dataset_idx, (start, end) in enumerate(self.dataset_ranges):
            if start <= idx < end:
                # Adjust the index to the dataset's local index
                local_idx = idx - start
                dataset = self.datasets[dataset_idx]
                
                # Fetch the image and label from the corresponding dataset
                image, label = dataset[local_idx]
                
                # Map the label to the new global label
                label = self.label_mapping[dataset_idx][label]

                # Apply the transform if specified for this dataset
                if self.transforms[dataset_idx] is not None:
                    image = self.transforms[dataset_idx](image)
                
                # If return 2 elements, return the image and the label
                return image, label
        
        # If index is out of bounds, raise an error
        raise IndexError(f"Index {idx} is out of bounds")
    
    def __getitem_extended__(self, idx):
        """
        Given an index, return the appropriate sample and the adjusted label.
        """
        # Find out which dataset the index belongs to
        for dataset_idx, (start, end) in enumerate(self.dataset_ranges):
            if start <= idx < end:
                # Adjust the index to the dataset's local index
                local_idx = idx - start
                dataset = self.datasets[dataset_idx]
                
                # Fetch the image and label from the corresponding dataset
                image, local_label = dataset[local_idx]
                
                # Map the label to the new global label
                label = self.label_mapping[dataset_idx][local_label]

                # Apply the transform if specified for this dataset
                if self.transforms[dataset_idx] is not None:
                    image = self.transforms[dataset_idx](image)
                
                return image, label, dataset_idx, local_label
            
        # If index is out of bounds, raise an error
        raise IndexError(f"Index {idx} is out of bounds")
        


class CombinedLabelDataset(Dataset):
    def __init__(self, datasets, transforms=None):
        """
        Args:
            datasets (list of torchvision datasets): A list of torchvision datasets to be merged.
        """
        self.datasets = datasets
        self.transforms = transforms if transforms is not None else [None] * len(datasets)
        
        # Combine the classes and update label indices for each dataset
        self.classes = []
        self.dataset_ranges = []
        offset = 0
        
        # In this class, we will have a single label for each of the datasets
        self.labels = []
        
        for d_idx, dataset in enumerate(tqdm(datasets, desc="Combining datasets")):
            # dataset_classes = get_classes(dataset)
            # Append the current dataset's classes to the global class list
            # self.classes.extend(dataset_classes)
            self.classes.append(d_idx)
            
            # Save the range of indices from this dataset
            start_idx = offset
            end_idx = offset + len(dataset)
            self.dataset_ranges.append((start_idx, end_idx))
            offset += len(dataset)
            
            # Add the label for this dataset samples using the dataset index
            self.labels.extend([d_idx] * len(dataset))

            tqdm.write(f"Added {len(dataset)} samples and 1 class from {type(dataset).__name__}")
        
        # Calculate total length
        self.total_length = sum(len(d) for d in datasets)
        print(f"Combined dataset has {len(self.classes)} classes and {self.total_length} samples.")

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """
        Given an index, return the appropriate sample and the adjusted label.
        """
        # Find out which dataset the index belongs to
        for dataset_idx, (start, end) in enumerate(self.dataset_ranges):
            if start <= idx < end:
                # Adjust the index to the dataset's local index
                local_idx = idx - start
                dataset = self.datasets[dataset_idx]
                
                # Fetch the image and label from the corresponding dataset
                image, label = dataset[local_idx]
                
                # Obtain the label for this dataset
                label = self.labels[idx]

                # Apply the transform if specified for this dataset
                if self.transforms[dataset_idx] is not None:
                    image = self.transforms[dataset_idx](image)
                
                return image, label
        
        # If index is out of bounds, raise an error
        raise IndexError(f"Index {idx} is out of bounds")


def VMMRdb(root, split='train', ood=False, div='bm', transform=None):
    """VMMRdb dataset from
    `"VMMRdb: A Multi-Modal Dataset for Vehicle Make and Model Recognition"
    <https://arxiv.org/abs/1905.02191>`_.

    Args:

        root (string): Root directory of dataset where directory
            ``VMMRdb`` exists.

        split (string, optional): The dataset split, supports ``train``, ``test``, ``val``.
            Default: ``train``.

        ood (bool, optional): If True, creates the out-of-distribution split of the dataset.
            Default: ``False``.

        div (string, optional): The dataset class division, supports ``b`` (brand), ``bm`` (brand-model) and ``bmy`` (brand-model-year).

    """
    split_folder = split
    div_folder = div
    d_folder = "ood" if ood else "id"

    root_folder = os.path.join(root, "vmmrdb", div_folder, d_folder, split_folder)
    return ImageFolder(root_folder, transform=transform)


def load_all_datasets(root, split='train', transforms=None):
    if transforms is None:
        transforms = [None] * 8
    if split == 'train':
        print("Loading CIFAR10 [1/8]")
        cifar10 = CIFAR10(root=root, train=True, download=False, transform=transforms[0])
        print("Loading CIFAR100 [2/8]")
        cifar100 = CIFAR100(root=root, train=True, download=False, transform=transforms[1])
        print("Loading EMNIST letters [3/8]")
        emnist_letters = EMNIST(root=root, split='letters', train=True, download=False, transform=transforms[2])
        print("Loading EMNIST digits [4/8]")
        emnist_digits = EMNIST(root=root, split='mnist', train=True, download=False, transform=transforms[3])
        print("Loading Places365 [5/8]")
        places = Places365(root=root, split='train-standard', download=False, small=True, transform=transforms[4])
        print("Loading SVHN [6/8]")
        svhn = SVHN(root=root, split='train', download=False, transform=transforms[5])
        print("Loading VMMRdb ID [7/8]")
        vmmrdb_id = VMMRdb(root=root, split='train', ood=False, div='bm', transform=transforms[6])
        print("Loading VMMRdb OOD [8/8]")
        vmmrdb_ood = VMMRdb(root=root, split='train', ood=True, div='bm', transform=transforms[7])
    elif split == 'test':
        print("Loading CIFAR10 [1/8]")
        cifar10 = CIFAR10(root=root, train=False, download=False, transform=transforms[0])
        print("Loading CIFAR100 [2/8]")
        cifar100 = CIFAR100(root=root, train=False, download=False, transform=transforms[1])
        print("Loading EMNIST letters [3/8]")
        emnist_letters = EMNIST(root=root, split='letters', train=False, download=False, transform=transforms[2])
        print("Loading EMNIST digits [4/8]")
        emnist_digits = EMNIST(root=root, split='mnist', train=False, download=False, transform=transforms[3])
        print("Loading Places365 [5/8]")
        places = Places365(root=root, split='val', download=False, small=True, transform=transforms[4])
        print("Loading SVHN [6/8]")
        svhn = SVHN(root=root, split='test', download=False, transform=transforms[5])
        print("Loading VMMRdb ID [7/8]")
        vmmrdb_id = VMMRdb(root=root, split='test', ood=False, div='bm', transform=transforms[6])
        print("Loading VMMRdb OOD [8/8]")
        vmmrdb_ood = VMMRdb(root=root, split='test', ood=True, div='bm', transform=transforms[7])

    return cifar10, cifar100, emnist_letters, emnist_digits, places, svhn, vmmrdb_id, vmmrdb_ood

def load_joint_dataset(root, split='train', transforms=None):
    cifar10, cifar100, emnist_letters, emnist_digits, places, svhn, vmmrdb_id, vmmrdb_ood = load_all_datasets(root, split)

    return CombinedDataset([cifar10, cifar100, emnist_letters, emnist_digits, places, svhn, vmmrdb_id, vmmrdb_ood], transforms=transforms)


def load_joint_labels_dataset(root, split='train', transforms=None):
    cifar10, cifar100, emnist_letters, emnist_digits, places, svhn, vmmrdb_id, vmmrdb_ood = load_all_datasets(root, split)

    return CombinedLabelDataset([cifar10, cifar100, emnist_letters, emnist_digits, places, svhn, vmmrdb_id, vmmrdb_ood], transforms=transforms)

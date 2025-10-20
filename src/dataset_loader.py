from torch.utils.data import Dataset

import numpy as np
from sklearn.utils import compute_class_weight


class ClassificationDataset(Dataset):


    def __init__(self, X, y, transforms=None):
        """
        Constructs a new dataset

        Args:
        """

        self.X = X
        self.y = y
        self.transforms = transforms
        self.len = len(self.X)


    def __len__(self):
        """
        Python operator to compute dataset length

        Returns: ``int``
            Dataset size
        """
        return self.len


    def __getitem__(self, idx):
        """
        Python operator to access one sample of the dataset from its position

        Args:
            idx: ``int``
                Sample position
        
        Returns: ``(PIL.Image, int)``
        """
        #print(f"{idx}_{self.len}_{len(self.X_2)}_{self.__len__()}", flush=True)
        # Get sample
        sample = self.y[idx]
        image = self.transforms(self.X[idx])

        return image, sample
    

    def compute_weights(self):
        return compute_class_weight("balanced", classes= np.unique(np.array(self.y)), y=np.array(self.y))
    
class OutlinerDataset(Dataset):


    def __init__(self, X, y, X_2 = None, y_2=None, transforms=None, transforms_2=None):
        """
        Constructs a new dataset

        Args:
        """

        self.X = X
        self.y = y
        self.X_2 = X_2
        self.y_2 = y_2
        self.transforms = transforms
        self.transforms_2 = transforms_2
        self.len = len(self.X)


    def __len__(self):
        """
        Python operator to compute dataset length

        Returns: ``int``
            Dataset size
        """
        if self.X_2 is not None:
            return self.len + len(self.X_2)
        else:
            return self.len


    def __getitem__(self, idx):
        """
        Python operator to access one sample of the dataset from its position

        Args:
            idx: ``int``
                Sample position
        
        Returns: ``(PIL.Image, int)``
        """
        #print(f"{idx}_{self.len}_{len(self.X_2)}_{self.__len__()}", flush=True)
        # Get sample
        if idx < self.len:
            sample = self.y[idx]
            image = self.transforms(self.X[idx])
            outlier = False
        else: 
            sample = self.y_2[idx - self.len]
            image = self.transforms_2(self.X_2[idx - self.len])
            outlier = True

        return image, sample, outlier
    

    def compute_weights(self):
        return compute_class_weight("balanced", classes= np.unique(np.array(self.y)), y=np.array(self.y))
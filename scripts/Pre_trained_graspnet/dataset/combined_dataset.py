import numpy as np
from torch.utils.data import Dataset
class ConcatDataset(Dataset):
    '''
    This class is combining different dataset into one datasets
    '''
    def __init__(self,dataset_list, probs=None):
        self.datasets = dataset_list
        self.dataset_types = ['graspnet', 'meta']  # can add more dataset types here in the list
        self.dataset_type =self.dataset_types[0]
        self.probs = probs if probs else [1/len(self.datasets)] * len(self.datasets)

    def __getitem__(self, i):
        # chooses the random dataset from the list
        index = np.random.randint(0, len(self.dataset_types))
        self.dataset_type = self.dataset_types[index]
        chosen_dataset = self.datasets[index]
        i = i % len(chosen_dataset)
        return chosen_dataset[i]

    def __len__(self):
        return  max(len(d) for d in self.datasets)
    
    def get_dataset_type(self):
        return self.dataset_type
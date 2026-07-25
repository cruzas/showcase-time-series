from torch.utils.data import Dataset

class WalmartSalesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Arguments:
            - csv_file (string): Path to the csv file with annotations.
            - root_dir (string): Directory with all the data.
            - transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
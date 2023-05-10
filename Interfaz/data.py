
import os, sys
from libs import *

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df_path, data_path, 
        config, 
        augment = False, 
    ):
        self.df_path, self.data_path,  = df_path, data_path, 
        self.df = pandas.read_csv(self.df_path)

        self.config = config
        self.augment = augment

    def __len__(self, 
    ):
        return len(self.df)

    def __getitem__(self, 
        index, 
    ):
        row = self.df.iloc[index]

        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        # call load_data with allow_pickle implicitly set to true
        ecg = np.load("{}/{}.npy".format(self.data_path, row["id"]))[self.config["ecg_leads"], :]

        # restore np.load for future normal usage
        np.load = np_load_old

        ecg = pad_sequences(ecg, self.config["ecg_length"], "float64", 
            "post", "post", 
        )
        if self.augment:
            ecg = self.drop_lead(ecg)
        ecg = torch.tensor(ecg).float()

        return ecg
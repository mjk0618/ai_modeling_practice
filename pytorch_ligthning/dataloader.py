import pandas as pd
import pytorch_lightning as pl
import torch
from dataset import MTDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

SEED = 42
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MTDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../data/",
        tokenizer: AutoTokenizer = AutoTokenizer,
        batch_size: int = 16,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage: str):
        data = pd.read_csv(self.data_dir)

        if stage == "fit": 
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=SEED)
            self.train_data = MTDataset(train_data, self.tokenizer)
            self.val_data = MTDataset(val_data, self.tokenizer)

        elif stage == "test":
            self.test_data = MTDataset(data, self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size)
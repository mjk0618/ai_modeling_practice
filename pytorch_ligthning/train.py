import os

import pytorch_lightning as pl
from dataloader import MTDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from model import MTNet

pl.seed_everything(42, workers=True)

MODEL_NAME = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


prj_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(
    prj_dir, "data", "train_data.csv"
)
dataloader = MTDataLoader(data_dir=dataset_dir, tokenizer=tokenizer, batch_size=32)
model = MTNet(model_name=MODEL_NAME, learning_rate=5e-5, weight_decay=0.01, warmup_steps=0)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"./{MODEL_NAME.split('/')[-1]}/full_data",
    monitor="val_bleu", 
    mode="max",
    save_top_k=2,
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    max_epochs=3,
)

trainer.fit(model=model, train_dataloaders=dataloader)
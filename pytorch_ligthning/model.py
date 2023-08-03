import pytorch_lightning as pl
import torch
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.functional import log_softmax
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)


class MTNet(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        weight_decay,
        warmup_steps,
        model_name="Helsinki-NLP/opus-mt-ko-en",
    ):
        super().__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def forward(self, batch):
        return self.model(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        return [optimizer], [scheduler]
    
    def total_steps(self):
        return len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
    
    def calculate_bleu(self, predictions, targets):
        bleu_score = 0.0
        for pred, target in zip(predictions, targets):
            pred_tokens = self.tokenizer.decode(pred, skip_special_tokens=True)
            target_tokens = self.tokenizer.decode(target, skip_special_tokens=True)
            bleu_score += sentence_bleu([target_tokens.split()], pred_tokens.split())

        return bleu_score / len(predictions)

    def training_step(self, batch):
        outputs = self(batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(log_softmax(logits, dim=-1), dim=-1)
        bleu = self.calculate_bleu(preds, batch["labels"])

        self.log("train_loss", loss)
        self.log("train_belu", bleu)

        return {"loss": loss, "bleu": bleu}
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(log_softmax(logits, dim=-1), dim=-1)
        bleu = self.calculate_bleu(preds, batch["labels"])

        self.log("val_loss", loss)
        self.log("val_bleu", bleu)

        return {"loss": loss, "bleu": bleu}
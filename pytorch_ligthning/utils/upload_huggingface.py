from model import MTNet
from transformers import PreTrainedModel


checkpoint_path = "../checkpoint/opus-mt-ko-en/full_data/epoch=2-step=89922.ckpt"
model = MTNet.load_from_checkpoint(checkpoint_path, learning_rate=5e-5, weight_decay=0.01, warmup_steps=0)

hf_model = model.model
save_directory = './hf_model'
PreTrainedModel.save_pretrained(hf_model, save_directory)

'''
To upload PyTorch Lightning checkpoint to HuggingFace, code above should be executed.
This code splits .ckpt to a few files that can be uploaded to and loaded from HuggingFace.

Additional work has to be done in your terminal
Install git large file storage to upload model checkpoint

Download sentencepiece model for target and source word, as the model is translation model.
Donwload tokenizer and vocab related json files

>>> git lfs install
>>> git lfs track "pytorch_model.bin"
>>> git add *
>>> git commit -am "commit message"
>>> git push

Login to HuggingFace with your ID and PW to verify you are the owner of model repository.

Check if commit has been succesfully pushed to your repository.
'''

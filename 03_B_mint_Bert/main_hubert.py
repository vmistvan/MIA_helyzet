# serializációhoz és visszatöltéshez praktikus tudásanyag
# https://huggingface.co/transformers/v1.2.0/serialization.html#serialization-best-practices


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt

# %matplotlib inline

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import transformers
from transformers import BertModel, BertConfig, BertForQuestionAnswering
from transformers import AutoModel, BertTokenizerFast, BertTokenizer
import pandas as pd

#verify that correct package version are available
print(torch.__version__)
print(transformers.__version__)
print("pandas version:",pd.__version__)

# print("flash version:",flash.__version__)
print("cuda version:",torch.cuda_version)
print("torch / cuda version:",torch.version.cuda)
print("cuda available:",torch.cuda.is_available())

# pretrain_model  = AutoModel.from_pretrained('.\\hubert_pytorch\\pytorch_model.bin', return_dict=False)
# state_dict = torch.load('.\\hubert_pytorch\\pytorch_model.bin')
# model.load_state_dict(state_dict)
# majd később
# model = BertForQuestionAnswering.from_pretrained('.\\hubert_pytorch')
# tokenizer = BertTokenizer.from_pretrained('.\\hubert_pytorch')



torch.set_float32_matmul_precision('medium')

hun_train = pd.read_csv(".\\hubert_pytorch\\train.tsv", sep='\t')
hun_test = pd.read_csv(".\\hubert_pytorch\\test.tsv", sep='\t')

hun_train = hun_train[hun_train['label'] != 'snopes']
hun_train = hun_train[['main_text','label']]
hun_train = hun_train.dropna(subset=['main_text', 'label'])

print (hun_train.head())

hun_test = hun_test[['main_text','label']]
hun_test = hun_test.dropna(subset=['main_text', 'label'])

# az állítások számra mappelése
hun_train['label'] = hun_train['label'].map({"true":0, "false":1, "unproven":2, "mixture":3, "vulgarity":4, "insult":5})
hun_test['label'] = hun_test['label'].map({"true":0, "false":1, "unproven":2, "mixture":3, "vulgarity":4, "insult":5})




class HunClassifier(pl.LightningModule):

    def __init__(self, max_seq_len=512, batch_size=128, learning_rate = 0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()

        # self.pretrain_model = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.pretrain_model  = AutoModel.from_pretrained('.\\hubert_pytorch\\', return_dict=False)
        self.pretrain_model.eval()
        for param in self.pretrain_model.parameters():
            param.requires_grad = False


        self.new_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,6),
            nn.LogSoftmax(dim=1)
        )

    def prepare_data(self):
        # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', return_dict=False)

        tokenizer = BertTokenizerFast.from_pretrained('.\\hubert_pytorch\\', return_dict=False)

        tokens_train = tokenizer.batch_encode_plus(
            hun_train["main_text"].tolist(),
            max_length = self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )

        tokens_test = tokenizer.batch_encode_plus(
            hun_test["main_text"].tolist(),
            max_length = self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )

        self.train_seq = torch.tensor(tokens_train['input_ids'])
        self.train_mask = torch.tensor(tokens_train['attention_mask'])
        self.train_y = torch.tensor(hun_train["label"].tolist())

        self.test_seq = torch.tensor(tokens_test['input_ids'])
        self.test_mask = torch.tensor(tokens_test['attention_mask'])
        self.test_y = torch.tensor(hun_test["label"].tolist())

    def forward(self, encode_id, mask):
        _, output= self.pretrain_model(encode_id, attention_mask=mask)
        output = self.new_layers(output)
        return output

    def train_dataloader(self):
      train_dataset = TensorDataset(self.train_seq, self.train_mask, self.train_y)
      self.train_dataloader_obj = DataLoader(train_dataset, batch_size=self.batch_size)
      return self.train_dataloader_obj


    def test_dataloader(self):
      test_dataset = TensorDataset(self.test_seq, self.test_mask, self.test_y)
      self.test_dataloader_obj = DataLoader(test_dataset, batch_size=self.batch_size)
      return self.test_dataloader_obj

    def training_step(self, batch, batch_idx):
      encode_id, mask, targets = batch
      outputs = self(encode_id, mask)
      preds = torch.argmax(outputs, dim=1)
      #print(targets)
      # https://saturncloud.io/blog/calculating-the-accuracy-of-pytorch-models-every-epoch/
      train_accuracy = accuracy(preds, targets)
      loss = self.loss(outputs, targets)
      self.log('train_accuracy', train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
      self.log('train_loss', loss, on_step=False, on_epoch=True)
      return {"loss":loss, 'train_accuracy': train_accuracy}

    def test_step(self, batch, batch_idx):
      encode_id, mask, targets = batch
      outputs = self.forward(encode_id, mask)
      preds = torch.argmax(outputs, dim=1)
      test_accuracy = accuracy(preds, targets)
      loss = self.loss(outputs, targets)
      return {"test_loss":loss, "test_accuracy":test_accuracy}

    def test_epoch_end(self, outputs):
      test_outs = []
      for test_out in outputs:
          out = test_out['test_accuracy']
          test_outs.append(out)
      total_test_accuracy = torch.stack(test_outs).mean()
      self.log('total_test_accuracy', total_test_accuracy, on_step=False, on_epoch=True)
      return total_test_accuracy

    def configure_optimizers(self):
      params = self.parameters()
      optimizer = optim.Adam(params=params, lr = self.learning_rate)
      return optimizer



model = HunClassifier()
# trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=30, gpus=-1)
trainer = pl.Trainer(max_epochs=200, enable_progress_bar=True, accelerator='gpu', devices=-1)
trainer.fit(model)

trainer.save_checkpoint("hubert_model.pt")

trainer.test()

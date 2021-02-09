import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from bert_data_loader import create_data_loader
from train_eval import train_epoch, eval_model

if __name__ == "__main__":
    df_train = pd.read_csv('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/train.csv')
    df_val = pd.read_csv('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/val.csv')
    df_test = pd.read_csv('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a4/data/test.csv')

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_data_loader = create_data_loader(df_train[:10000], tokenizer, 160, 16)
    val_data_loader = create_data_loader(df_val, tokenizer, 160, 16)
    test_data_loader = create_data_loader(df_test, tokenizer, 160, 16)

    class SentimentClassifier(nn.Module):
        def __init__(self, n_classes):
            super(SentimentClassifier, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.drop = nn.Dropout(p=0.3)
            self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

        def forward(self, input_ids, attention_mask):
            _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
            )
            output = self.drop(pooled_output)
            return self.out(output)

    EPOCHS = 10
    model = SentimentClassifier(5)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):      
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            scheduler,
            len(df_train)
        )
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            len(df_val)
        )

from typing import List
import torch
import numpy as np
import pandas as pd
from roberta import RobertaDataset
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
#from tqdm import tqdm
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)
from earlystopping import EarlyStopping

class Classifier:
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
        self.collate_fn = False
        self.label_encoder = None

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        # import train dev data
        train_data = pd.read_csv(train_filename, sep='\t', names=['label', 'category', 'target', 'offset', 'text'])
        dev_data = pd.read_csv(dev_filename, sep='\t', names=['label', 'category', 'target', 'offset', 'text'])

        # preprocess train dev data
        train_dataset = RobertaDataset(train_data)
        dev_dataset = RobertaDataset(dev_data)

        # prepare the dataloader
        self.collate_fn = train_dataset.collate_fn
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=train_dataset.collate_fn)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)

        self.model.to(device)
        early_stopping = EarlyStopping(patience=3, verbose=False)

        num_epoch = 100
        for epoch in range(num_epoch):
            # training process
            self.model.train()
            total_loss_train = []
            total_acc_train = []
            #progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')

            for batch in train_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss_train.append(loss.item())

                predictions = logits.argmax(dim=1)
                total_correct = (predictions == labels).sum().item()
                total_samples = labels.size(0)
                batch_accuracy = total_correct / total_samples
                total_acc_train.append(batch_accuracy)
                #progress_bar.set_postfix({'Accuracy': batch_accuracy})

            #print(f"Epoch {epoch + 1}, Train Loss: {sum(total_loss_train) / len(total_loss_train)}, Train Accuracy: {sum(total_acc_train) / len(total_acc_train)}")

            # validation process
            self.model.eval()
            total_loss_val = []
            total_acc_val = []
            #val_progress_bar = tqdm(dev_loader, desc=f'Validation Epoch {epoch + 1}')

            with torch.no_grad():
                for batch in dev_loader:
                    inputs = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)

                    outputs = self.model(inputs)
                    logits = outputs.logits
                    batch_val_loss = criterion(logits, labels)
                    total_loss_val.append(batch_val_loss.item())

                    val_predictions = logits.argmax(dim=1)
                    val_total_correct = (val_predictions == labels).sum().item()
                    val_total_samples = labels.size(0)
                    val_accuracy = val_total_correct / val_total_samples
                    total_acc_val.append(val_accuracy)

                    #val_progress_bar.set_postfix({'Accuracy': val_accuracy})

            avg_valid_loss = sum(total_loss_val) / len(total_loss_val)
            #print(f"Epoch {epoch + 1}, Val Loss: {sum(total_loss_val) / len(total_loss_val)}, Val Accuracy: {sum(total_acc_val) / len(total_acc_val)}")

            # check the validation loss and implement early stopping
            early_stopping(avg_valid_loss, self.model)
            if early_stopping.early_stop:
                #print("Early stopping")
                break

        # reload the best model
        self.model.load_state_dict(torch.load('checkpoint.pt'))
        self.label_encoder = train_dataset.label_encoder

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        # load data and prepare the dataloader
        test_data = pd.read_csv(data_filename, sep='\t', names=['label', 'category', 'target', 'offset', 'text'])
        test_dataset = RobertaDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=self.collate_fn)

        self.model.to(device)
        self.model.eval()

        all_predictions = []

        # predicting process
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input_ids'].to(device)
                outputs = self.model(inputs)
                logits = outputs.logits

                predictions = logits.argmax(dim=1)
                all_predictions.extend(predictions.tolist())

        # decode the encoded label
        all_predictions = self.label_encoder.inverse_transform(all_predictions)

        return all_predictions


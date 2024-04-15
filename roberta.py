import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class RobertaDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # prepare each column
        label = self.data['label'][idx]
        text = self.data['text'][idx]
        aspect = self.data['category'][idx]
        target = self.data['target'][idx]
        offset = self.data['offset'][idx]
        
        # Extract start and end positions from offset
        if ':' in offset:
            start, end = map(int, offset.split(':'))
        else:
            start, end = 0, 0

        # combine text, target and aspect with special token
        input_text = f"{text[:start]}[SEP]{target}[SEP]{text[end:]}[SEP]{aspect}"
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        # prepare attention_mask (before padding)
        attention_mask = [1] * len(input_ids)
        # encoding labels
        label = self.label_encoder.transform([label])[0]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
	 }

    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]  # Extract attention masks
        labels = [item['label'] for item in batch]

        # Pad sequences to the same length
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(labels, dtype=torch.long)
        }


import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import os
from functools import partial

class MathTextbookDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_length=512):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.eos_id = self.tokenizer.token_to_id("[SEP]")

        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        encoded = self.tokenizer.encode(text)
        token_ids = encoded.ids

        token_ids = token_ids + [self.eos_id]

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]

        return torch.tensor(token_ids, dtype=torch.long)

def math_collate_fn(batch, pad_id):
    max_len = max(len(item) for item in batch)
    
    batch_inputs = []
    batch_targets = []
    
    for item in batch:
        pad_len = max_len - len(item)
        
        if pad_len > 0:
            pads = torch.full((pad_len,), pad_id, dtype=torch.long)
            padded_item = torch.cat([item, pads])
        else:
            padded_item = item
            
        batch_inputs.append(padded_item[:-1])
        batch_targets.append(padded_item[1:])
        
    return torch.stack(batch_inputs), torch.stack(batch_targets)


def get_math_dataloader(file_path, tokenizer_path, max_length=128, batch_size=8):
    from functools import partial
    
    dataset = MathTextbookDataset(file_path, tokenizer_path, max_length)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(math_collate_fn, pad_id=dataset.pad_id),
        num_workers=2,
        pin_memory=True
    )
    
    return dataset, dataloader

# 这样如果你还要单独测试 data_loader.py，可以保留这段：
if __name__ == "__main__":
    test_ds, test_dl = get_math_dataloader(
        "../data/processed_latex/train_data_100_all.mmd", 
        "../data/tokenizer_math.json"
    )
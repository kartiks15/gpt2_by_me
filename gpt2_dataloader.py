import tiktoken
import importlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# tokenizer = tiktoken.get_encoding("gpt2")

# Example usage
# text = "Hello, how are you? You are doing great!"
# encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>", "<|unk|>"})
# print("Encoded:", encoded)

# decoded = tokenizer.decode(encoded)
# print("Decoded:", decoded)

# with open ("train_data.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# enc_text = tokenizer.encode(raw_text)
# print("total no of chars:", len(raw_text))

class GPT2Dataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    
def dataloader_v1(text, batch_size, max_length, stride, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPT2Dataset(text, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

with open("train_data.txt", "r", encoding = "utf-8") as f:
        raw_text = f.read()
    
vocab_size = 50257
out_dim = 256
context_length = 1024

token_embedding_layer = torch.nn.Embedding(vocab_size, out_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, out_dim)

batch_size = 8
max_len = 4

dataloader = dataloader_v1(raw_text, batch_size=batch_size, max_length = max_len, stride = max_len)

for batch in dataloader:
     x, y = batch
     token_embedding = token_embedding_layer(x)
     pos_embedding = pos_embedding_layer(torch.arange(max_len))

     input_embedding = token_embedding+pos_embedding

     break

print(input_embedding.shape)
import pandas as pd
import selfies as sf
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import os
import json
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles


# fixes some conda issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, input_len):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_len, 256),
                                nn.ReLU(),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Linear(64, 16),
                                nn.ReLU(),
                                nn.Linear(16, 1))

    def forward(self, x):
        return self.fc(x)


def logp(smiles):
    return MolLogP(MolFromSmiles(smiles))


def preprocess_and_save_data(file, data_dir):
    df = pd.read_csv(file, sep='\t')
    df['SELFIES'] = df['SMILES'].apply(sf.encoder)
    vocab = list(sorted(sf.get_alphabet_from_selfies(df['SELFIES']))) + ['[nop]']
    symbol_to_idx = {symbol: i for i, symbol in enumerate(vocab)}
    idx_to_symbol = {i: symbol for i, symbol in enumerate(vocab)}
    max_len = df['SELFIES'].apply(sf.len_selfies).max()
    df['encoded'] = df['SELFIES'].apply(sf.selfies_to_encoding, args=(symbol_to_idx, max_len, 'one_hot'))
    x = torch.tensor(np.vstack(df['encoded'].apply(lambda x: np.array(x).flatten())), dtype=torch.float)
    df['logP'] = df['SMILES'].apply(logp)
    y = torch.tensor(df['logP'], dtype=torch.float).view((-1, 1))
    torch.save(x, data_dir + '/x.pt')
    torch.save(y, data_dir + '/y.pt')
    json.dump({'symbol_to_idx': symbol_to_idx, 'idx_to_symbol': idx_to_symbol, 'max_len': int(max_len)}, open(data_dir + '/vocab.json', 'w'))


def load_data(data_dir):
    x = torch.load(data_dir + '/x.pt').to('cuda')
    y = torch.load(data_dir + '/y.pt').to('cuda')
    vocab = json.load(open(data_dir + '/vocab.json', 'r'))
    symbol_to_idx, idx_to_symbol, max_len = vocab['symbol_to_idx'], vocab['idx_to_symbol'], vocab['max_len']
    idx_to_symbol = {int(key): idx_to_symbol[key] for key in idx_to_symbol}
    dataset = TensorDataset(x, y)
    train_data, test_data = random_split(dataset, [int(round(len(dataset) * 0.8)), int(round(len(dataset) * 0.2))])
    train_dataloader, test_dataloader = DataLoader(train_data, batch_size=10000, shuffle=True), DataLoader(test_data, batch_size=10000)
    return train_dataloader, test_dataloader, symbol_to_idx, idx_to_symbol, max_len


def train(model, train_dataloader, test_dataloader):
    for epoch in range(5):
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            loss = loss_f(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            total_loss = 0
            for x_batch, y_batch in test_dataloader:
                total_loss += loss_f(model(x_batch), y_batch).item()
            print(total_loss)
    return model


def indices_to_smiles(indices, idx_to_symbol):
    selfies = ''.join([idx_to_symbol[idx] for idx in indices])
    return sf.decoder(selfies)


def dream(model, starting_one_hot):
    old_smiles = ''
    in_selfies = starting_one_hot
    in_selfies += torch.rand(in_selfies.shape, device='cuda') * 0.5
    in_selfies[in_selfies > 1] = 1
    in_selfies = in_selfies.clone().detach().view((1, -1)).requires_grad_(True)
    reverse_optimizer = optim.Adam([in_selfies], lr=0.01)
    vals = []
    for epoch in range(10000):
        reverse_optimizer.zero_grad()
        out = model(in_selfies)
        indices = in_selfies.detach().view((max_len, -1)).argmax(dim=1).tolist()
        smiles = indices_to_smiles(indices, idx_to_symbol)
        if smiles != old_smiles:
            print(f"New molecule: logP: {logp(smiles)}, SMILES: {smiles}")
            vals.append(logp(smiles))
            old_smiles = smiles
        out.backward()
        reverse_optimizer.step()
    return old_smiles, vals


if __name__ == '__main__':
    # preprocess_and_save_data('gdb11_size09.smi', 'data')
    train_dataloader, test_dataloader, symbol_to_idx, idx_to_symbol, max_len = load_data('data')
    model = Net(max_len * len(symbol_to_idx)).to('cuda')
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model = train(model, train_dataloader, test_dataloader)

    x_batch, y_batch = next(iter(test_dataloader))
    plt.scatter(model(x_batch).detach().cpu(), y_batch.detach().cpu())
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.show()

    improvement_count = 0
    from tqdm import tqdm
    for i in tqdm(range(100)):
        final_smiles, vals = dream(model, x_batch[2])
        improvement_count += int(vals[0] > vals[-1])
    print(improvement_count / 100)
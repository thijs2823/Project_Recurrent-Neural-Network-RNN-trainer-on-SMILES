import json
import numpy as np
from rdkit import Chem
from tensorflow.keras.models import load_model

# Load model
model_path = r"smiles_rnn_tf.h5"
model = load_model(model_path)

# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer = json.load(f)
stoi = tokenizer["stoi"]
itos = {int(k): v for k, v in tokenizer["itos"].items()}  # JSON keys zijn strings
vocab_size = len(stoi) + 1
max_len = 120

# Encode, decode
def encode(smiles):
    seq = [stoi[ch] for ch in smiles if ch in stoi]
    return seq[:max_len] + [0]*(max_len - len(seq))

def decode(seq):
    return "".join([itos[i] for i in seq if i in itos and i != 0])

# Generator
def generate(model, start="C", max_len=120, temperature=1.0, n_attempts=5):
    for _ in range(n_attempts):
        seq = [stoi.get(ch, 1) for ch in start if ch in stoi]
        for _ in range(max_len - len(seq)):
            x_pred = np.array(seq).reshape(1, -1)
            preds = model.predict(x_pred, verbose=0)[0, -1]
            preds = np.log(preds + 1e-8) / temperature
            exp_preds = np.exp(preds)
            probs = exp_preds / np.sum(exp_preds)
            top_idx = np.random.choice(len(probs), p=probs)
            if top_idx == 0:
                break
            seq.append(top_idx)
        smiles = decode(seq)
        if Chem.MolFromSmiles(smiles):
            return smiles
    return None


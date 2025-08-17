import numpy as np
from rdkit import Chem
from tensorflow.keras.models import load_model

# Load model
model = load_model("smiles_rnn_tf.h5")

# Load tokenizer
from tokenizer import stoi, itos
vocab_size = len(stoi) + 1

def decode(seq):
    return "".join([itos[i] for i in seq if i > 0])

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


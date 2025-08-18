import os, json, ast
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Loading data from CSV
csv_path = os.path.join(os.path.dirname(__file__), "approved_drugs_full.csv")
df_all = pd.read_csv(csv_path)

# 2. Data cleaning
if 'canonical_smiles' in df_all.columns:
    df_all['canonical_smiles'] = df_all['canonical_smiles'].fillna("").str.strip()
else:
    def extract_smiles(x):
        if not isinstance(x, str):
            return ""
        try:
            data = ast.literal_eval(x)
            if isinstance(data, dict):
                return data.get('canonical_smiles', "")
        except (ValueError, SyntaxError):
            return ""
        return ""
    if 'molecule_structures' in df_all.columns:
        df_all['canonical_smiles'] = df_all['molecule_structures'].fillna('').apply(extract_smiles).str.strip()
    else:
        df_all['canonical_smiles'] = ""

df_clean = df_all[df_all['canonical_smiles'].str.strip() != ''].copy()

# Creates a cleaned DataFrame of molecule names and their canonical SMILES strings
smiles_list = df_clean['canonical_smiles'].tolist()
names = df_clean.get('pref_name', pd.Series(["Unknown"]*len(df_clean))).fillna("Unknown").tolist()
print(len(smiles_list))

df = pd.DataFrame({"Name": names, "SMILES": smiles_list})

# 3. Tokenizer
alphabet = sorted(list(set("".join(smiles_list))))
stoi = {ch: i+1 for i, ch in enumerate(alphabet)}  # 0 = padding
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
max_len = 120
seq_len = max_len - 1

def encode(smiles):
    seq = [stoi[ch] for ch in smiles if ch in stoi]
    return seq[:max_len] + [0]*(max_len - len(seq))

def decode(seq):
    return "".join([itos[i] for i in seq if i > 0])

# Saving tokenizer
with open("tokenizer.json", "w") as f:
    json.dump({"stoi": stoi, "itos": itos}, f)

# 4. Preperation of the dataset
X = []
y = []
for s in smiles_list:
    seq = encode(s)
    X.append(seq[:-1])
    y.append(seq[1:])

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

# Train, validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# 5. Model
model = models.Sequential([
    layers.Input(shape=(seq_len,)),
    layers.Embedding(input_dim=vocab_size, output_dim=256),
    layers.LSTM(256, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(256, return_sequences=True),
    layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 6. Training phase
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=50
)

# 7. Saving model
model.save("smiles_rnn_tf.h5")

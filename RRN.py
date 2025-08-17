import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Loading data from CSV
csv_path = os.path.join(os.path.dirname(__file__), "approved_drugs_full.csv")
df_all = pd.read_csv(csv_path)

# 2. Data cleaning
if 'canonical_smiles' in df_all.columns:
    df_all['canonical_smiles'] = df_all['canonical_smiles'].fillna("").str.strip()
else:
    import ast
    def extract_smiles(x):
        try:
            data = ast.literal_eval(x)
            return data.get('canonical_smiles', "")
        except:
            return ""
    df_all['canonical_smiles'] = df_all['molecule_structures'].apply(extract_smiles).str.strip()

# Creates a cleaned DataFrame of molecule names and their canonical SMILES strings
smiles_list = [s for s in df_all['canonical_smiles'].tolist() if s]
names = df_all.get('pref_name', pd.Series(["Unknown"]*len(df_all))).fillna("Unknown").tolist()
print(len(smiles_list))

df = pd.DataFrame({"Name": names, "SMILES": smiles_list})

# 3. Tokenizer
alphabet = sorted(list(set("".join(smiles_list))))
stoi = {ch: i+1 for i, ch in enumerate(alphabet)}  # 0 = padding
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
max_len = 120

def encode(smiles):
    seq = [stoi[ch] for ch in smiles if ch in stoi]
    return seq[:max_len] + [0]*(max_len - len(seq))

def decode(seq):
    return "".join([itos[i] for i in seq if i > 0])

# 4. Preperation of the dataset
X = []
y = []
for s in smiles_list:
    seq = encode(s)
    X.append(seq[:-1])
    y.append(seq[1:])

X = np.array(X)
y = np.array(y)
y_onehot = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# 4. Model
model = models.Sequential([
    layers.Embedding(vocab_size, 256, input_length=max_len-1),
    layers.LSTM(256, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(256, return_sequences=True),
    layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# 5. Training phase 
model.fit(X, y_onehot, batch_size=64, epochs=50)

# 6. Saving of model
model.save("smiles_rnn_tf.h5")

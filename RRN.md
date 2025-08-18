## Importing necessary libraries
```python
import os, json, ast
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
```

## 1. Loading data from CSV
#### Loads data and reads CSV as Pandas DataFrame
```python
csv_path = os.path.join(os.path.dirname(__file__), "approved_drugs_full.csv")
df_all = pd.read_csv(csv_path)
```

## 2. Data cleaning
#### The canonical_smiles column is cleaned by replacing missing values with empty strings and stripping whitespace; only rows with non-empty SMILES are retained in df_clean:
```python
df_all['canonical_smiles'] = df_all['canonical_smiles'].fillna("").str.strip() # Removes 
df_clean = df_all[df_all['canonical_smiles'].str.strip() != ''].copy()
```
#### Converts the cleaned SMILES and molecule names into Python lists and combines them into a Pandas DataFrame.
```python
smiles_list = df_clean['canonical_smiles'].tolist()
names = df_clean.get('pref_name', pd.Series(["Unknown"]*len(df_clean))).fillna("Unknown").tolist()
print(len(smiles_list))

df = pd.DataFrame({"Name": names, "SMILES": smiles_list})
```

## 3. Tokenizer
#### All unique characters are converted into a sorted list and assigned an index. Index 0 is reserved for padding. The stoi mapping (string-to-index) converts characters to their corresponding indices, while itos (index-to-string) performs the reverse mapping.
```python
alphabet = sorted(list(set("".join(smiles_list))))
stoi = {ch: i+1 for i, ch in enumerate(alphabet)} 
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
max_len = 120
seq_len = max_len - 1

def encode(smiles):
    seq = [stoi[ch] for ch in smiles if ch in stoi]
    return seq[:max_len] + [0]*(max_len - len(seq))

def decode(seq):
    return "".join([itos[i] for i in seq if i > 0])
```

### Saving tokenizer
#### Saving tokinizer to JSON for later use
```python
with open("tokenizer.json", "w") as f:
    json.dump({"stoi": stoi, "itos": itos}, f)
```

## 4. Preperation of the dataset
#### For each SMILES string, the input consists of all characters except the last one, and the output consists of all characters except the first one. This follows a next-token prediction training setup.
```python
X = []
y = []
for s in smiles_list:
    seq = encode(s)
    X.append(seq[:-1])
    y.append(seq[1:])
```

#### Converts X and Y to numpy arrays
```python
X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)
```

#### Train, validation split (90% training, 10% validation)
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
```

## 5. Model
#### The model architecture consists of: Input → Embedding → LSTM → Dropout → LSTM → Dense with softmax activation. It predicts the next character at each timestep.
```python
model = models.Sequential([
    layers.Input(shape=(seq_len,)),
    layers.Embedding(input_dim=vocab_size, output_dim=256),
    layers.LSTM(256, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(256, return_sequences=True),
    layers.Dense(vocab_size, activation='softmax')
])
```

#### Compiled with Adam optimizer and categorical cross-entropy loss (predicted distribution vs. true index).
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

## 6. Training phase
#### Trained 50 epochs, batch size 64, with validation monitoring.
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=50
)
```
## 7. Saving model as HDF5 file
```python
model.save("smiles_rnn_tf.h5")
```

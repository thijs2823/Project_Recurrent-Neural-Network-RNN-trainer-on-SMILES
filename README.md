# Recurrent-Neural-Network-RNN-trainer-on-SMILES
This project aims to train a Recurrent Neural Network (RNN)  on SMILES representations of molecular structures of known pharmaceuticals.

## Dataset
- The model loads 3,594 approved drugs from the ChEMBL database (https://www.ebi.ac.uk/chembl/) (search for approved drugs).
- Each molecule is represented by its canonical SMILES string with its preferred name.

## Model
- TensorFlow
- Sequential RNN with Embedding + LSTM layers.
- Trained using categorical cross-entropy on tokenized SMILES sequences.
- Saves as `smiles_rnn_tf.h5`

## Requirements
- Python 3.10+
- TensorFlow 2.x
- pandas
- numpy

## Training
- Batch size: 64
- Epochs: 50
- Loss function: Categorical cross-entropy
- Optimizer: Adam

## Sample training curve (Epoch vs Loss):

<img width="470" height="557" alt="image" src="https://github.com/user-attachments/assets/d6a638f9-aca4-453d-9dbd-1881a82d9d6c" />

## Examples of generated drug-like SMILES
C[C@]12CC[C@@H]3c4ccc(OC(=O)N(CCCl)CCCl)cc4CC[C@H]3[C@@H]1CC[C@@H]2OP(=O)([O-])[O-].[Na+].[Na+]

<img width="520" height="308" alt="image" src="https://github.com/user-attachments/assets/9ef8b52e-61d8-4309-89e4-c83bee87bf52" />

CN1CCN(C2=Nc3ccccc3Oc3ccc(Cl)cc32)CC1.Cl

<img width="400" height="249" alt="image" src="https://github.com/user-attachments/assets/44d183c8-2598-49ca-a1ad-d0248b22d8b1" />


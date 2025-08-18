# Recurrent nural network (RNN) trainer on SMILES
This project aims to train a Recurrent Neural Network (RNN)  on SMILES representations of molecular structures of known pharmaceuticals.

## Dataset
- The model loads 4,194 approved drugs from the ChEMBL database (https://www.ebi.ac.uk/chembl/) (search for approved drugs).
- Each molecule is represented by its canonical SMILES string with its preferred name.
- Faulty records were removed manually

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
- The model is learning effectively from the training data, reaching a training loss of approximately 0.17 at epoch 50 (blue line).
- The validation loss begins to plateau around epoch 28.
- No signs of overfitting

<img width="680" height="600" alt="Figure" src="https://github.com/user-attachments/assets/4e059fdf-3a82-47bf-97de-5443b4f6533c" />

## Examples of generated drug-like SMILES
#### Cl.O=P(O)(O)C(O)(Cc1ccccc1)c1ccc(F)cc1F.Cl

<img width="230" height="298" alt="image" src="https://github.com/user-attachments/assets/2a74d4d2-c976-4a66-b2d1-5ae9810c62b2" />


#### CN1CCCC(OC(c2ccccc2)c2ccccc2)CC1

<img width="210" height="294" alt="image" src="https://github.com/user-attachments/assets/c59077b1-636f-4c8b-bce1-4df27e7463c7" />

#### Cc1cc(O)c(O)cc1OC

<img width="170" height="133" alt="image" src="https://github.com/user-attachments/assets/cb44f4a5-d4c0-4fa1-b897-1299952aa03e" />

#### CC1CNCCN1

<img width="120" height="109" alt="image" src="https://github.com/user-attachments/assets/28e0ff14-27ff-4a5b-a36c-d94e24282311" />



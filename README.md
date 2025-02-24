# RNN-Based Text Generation Project
This repository contains three assignments from the Neural Networks for Data Science Applications course, focusing on character-level language modeling using JAX. The project demonstrates the entire pipeline—ranging from RNN training to various text generation strategies.

Project Overview
Assignment 1: RNN Training
Implements a recurrent neural network (RNN) in pure JAX.
Trains the model on a text dataset (e.g., Penn Treebank) in a next-character prediction setting.
Visualizes training/validation losses and perplexities to confirm convergence.

Assignment 2: Text Generation (Sampling)
Autoregressive text generation using a sampling-based approach.
Takes an initial prompt, “warms up” the hidden state, then samples characters token by token.
Demonstrates how temperature scaling can yield more or less creative outputs.

Assignment 3: Beam Search
Implements a fully JAX-based beam search decoder.
Uses lax.scan to avoid explicit Python loops and maintains a fixed-size buffer for partial sequences.
Compares the resulting text with greedy decoding to show how beam search can produce more coherent (though sometimes repetitive) outputs.

## 📚 **Project Structure**
📂 Project Root   
├── [📄 README.md (This File)](README.md)    
├── [📊 AGE_REGRESSION.csv (Dataset)](AGE_PREDICTION.csv)    
└── [📒 run_MLP.ipynb (Jupyter Notebook for Training and Running the Model)](run_MLP.ipynb)  
- `assignment1_rnn_training.ipynb`: The RNN model creation and training loop.
- `assignment2_text_generation.py`: Sampling-based text generator.
- `assignment3_beam_search.py`: Beam search code plus comparisons to greedy decoding.
- `utils/`: (Optional) Utility scripts for data loading or pre-processing.
- `README.md`: This file.

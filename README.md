# Neural Network Forward Pass with NumPy Embeddings

## Overview

This project implements a simplified neural network forward pass using NumPy. The input is represented as a matrix of token IDs similar to how tokenized text can be represented before being processed by neural models.

The project includes token embedding lookup, padding handling, masked average pooling, linear transformations, ReLU activation and softmax probability calculation.

## What the Project Does

1. Creates a matrix of token IDs.
2. Adds padding tokens using the value 0.
3. Creates an embedding matrix with a zero row for padding.
4. Converts token IDs into embedding vectors.
5. Uses a padding mask to ignore padded positions.
6. Computes an average embedding for each sequence.
7. Applies linear transformations and ReLU activation.
8. Applies softmax to produce class probabilities.

## Technologies Used

- Python
- NumPy
- Embeddings
- Masking
- ReLU
- Softmax
- Deep Learning fundamentals

## What I Learned

Through this project, I practiced how tokenized input can be transformed into embeddings and processed through a simplified neural network forward pass. I also improved my understanding of padding masks, matrix multiplication, activation functions and probability generation with softmax.

## Future Improvements

- Add labels and cross-entropy loss.
- Calculate prediction accuracy.
- Turn the forward pass into a small trainable neural network.

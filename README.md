# Multilayer Perceptron from Scratch in C

A complete neural network implementation built from the ground up in C with no external libraries — only the C standard library. Developed for the Computational Intelligence course at the University of Ioannina (cse.uoi.gr).

## Network Architecture

```
Input (2)  ──>  Hidden 1 (30)  ──>  Hidden 2 (30)  ──>  Hidden 3 (30)  ──>  Output (3)
   x1, x2        Logistic             Logistic            Logistic           Logistic
                  /tanh/ReLU           /tanh/ReLU          /tanh/ReLU         (always)
```

**Total parameters:** 2,043 weights

The network classifies 2D points into 3 categories using one-hot encoding. The classification boundaries are defined by four circles centered at (+-0.5, +-0.5) with radius sqrt(0.2), split into upper/lower halves (C1/C2), with everything outside classified as C3.

## Configuration

All hyperparameters are compile-time constants in `src/mlp.c`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | 2 | Input dimensions |
| `p` | 3 | Output classes |
| `H1, H2, H3` | 30 | Neurons per hidden layer |
| `f` | 0 | Activation function (0=logistic, 1=tanh, 2=ReLU) |
| `n` | 0.2 | Learning rate |
| `B` | 40 | Mini-batch size (1=SGD, 4000=batch) |
| `EPOCHS` | 1000 | Maximum training epochs |
| `N` | 4000 | Samples per dataset |

## Building

Requires a C11 compiler and CMake 3.24+.

**With CMake:**
```bash
cmake -B build
cmake --build build
./build/mlp
```

**With gcc directly:**
```bash
gcc -std=c11 -O2 src/mlp.c -o mlp -lm
./mlp
```

To regenerate the training and test datasets:
```bash
gcc -std=c11 src/vectors.c -o vectors
./vectors
```

> **Note:** Run all binaries from the project root directory — the program reads from `data/` and writes output files to the working directory.

## Output

Training produces three output files:

- `train_errors.txt` — Error per epoch (CSV), used by the visualization notebook
- `correct_guesses.txt` — Correctly classified test points (CSV)
- `wrong_guesses.txt` — Misclassified test points (CSV)

Example output:
```
-----------------Training-----------------
---------------Training Done--------------
C1: 1188 guessed correctly out of 1225
C2: 1219 guessed correctly out of 1264
C3: 1491 guessed correctly out of 1511
Incorrect guesses: 102
Error percentage: 2.55%
Accuracy: 97.45%
```

## Visualization

Open `visualization.ipynb` in Jupyter to plot the training error curve and classification results. Requires Python with pandas, seaborn, and matplotlib.

## Project Structure

```
.
├── src/
│   ├── mlp.c              Main MLP implementation
│   └── vectors.c          Dataset generator
├── data/
│   ├── train_vectors.txt  Training set (4000 points)
│   └── test_vectors.txt   Test set (4000 points)
├── CMakeLists.txt
├── visualization.ipynb
├── LICENSE
└── README.md
```

## Implementation Details

- **Forward pass:** Computes weighted sums and applies activation functions layer by layer. The output layer always uses the logistic sigmoid regardless of the hidden layer activation.
- **Backpropagation:** Computes deltas from output to input using the chain rule, then calculates per-weight error derivatives.
- **Mini-batch gradient descent:** Accumulates gradients over B samples, averages them, and updates weights. Supports stochastic (B=1), mini-batch, and full-batch (B=N) modes.
- **Early stopping:** Training halts if the global error drops below 0.07 after epoch 700.

## License

See [LICENSE](LICENSE) for details.

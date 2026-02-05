# Machine Learning Assignment

**Name:** S.M. Nabiul Islam  
**Course:** Machine Learning  

---
*Readme is generated using AI*
---

## 📌 Objective

This assignment explores fundamental and advanced machine learning techniques, covering neural networks, sequence models, unsupervised learning, and generative AI. The goal is to implement, train, evaluate, and analyze various ML models using real-world and synthetic datasets.

---

## 📂 Project Structure

```
Machine-Learning-Assignment/
├── 0112230261.ipynb          # Main Jupyter notebook with all implementations
├── README.md                 # Project documentation
└── data/
    └── FashionMNIST/         # Fashion-MNIST dataset (auto-downloaded)
        └── raw/
            ├── t10k-images-idx3-ubyte
            ├── t10k-labels-idx1-ubyte
            ├── train-images-idx3-ubyte
            └── train-labels-idx1-ubyte
```

---

## 🛠️ Technologies & Libraries Used

- **Python 3.x**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning utilities
- **Transformers (Hugging Face)** - Pre-trained language models

---

## 📋 Tasks Completed

### Part 1: Neural Networks & Deep Learning

#### Task 1: Multilayer Perceptron (MLP) - Fashion-MNIST Classification

**Objective:** Build dense neural networks to classify images from the Fashion-MNIST dataset.

**What I Did:**
- Loaded and preprocessed the Fashion-MNIST dataset (70,000 grayscale images, 10 classes)
- Implemented 4 different MLP architectures:
  1. **Shallow MLP with ReLU** (784 → 256 → 128 → 10)
  2. **Shallow MLP with Sigmoid** (784 → 256 → 128 → 10)
  3. **Deep MLP with ReLU** (784 → 512 → 256 → 128 → 64 → 10)
  4. **Deep MLP with Sigmoid** (784 → 512 → 256 → 128 → 64 → 10)
- Trained all models with dropout regularization (0.2)
- Compared training/validation loss curves and test accuracy

**Key Achievements:**
- ✅ ReLU activation consistently outperformed Sigmoid
- ✅ Demonstrated the vanishing gradient problem with Sigmoid in deep networks
- ✅ Shallow MLP with ReLU provided the best balance of performance and efficiency

---

#### Task 2: Recurrent Neural Network (RNN/LSTM) - Time Series Prediction

**Objective:** Predict future values of a time series using LSTM and Simple RNN.

**What I Did:**
- Generated synthetic time series data (combination of sine waves + noise)
- Implemented sequence-to-value prediction using:
  1. **LSTM Model** (2 layers, 64 hidden units)
  2. **Simple RNN Model** for comparison
- Used a look-back window of 10 time steps
- Evaluated models using RMSE and MAE metrics

**Key Achievements:**
- ✅ LSTM achieved lower prediction error than Simple RNN
- ✅ Successfully captured both short-term and long-term patterns
- ✅ Visualized actual vs predicted values for model validation

---

#### Task 3: Self-Organizing Map (SOM) - Data Clustering

**Objective:** Map high-dimensional data into a 2D grid for visualization and clustering.

**What I Did:**
- Implemented a Self-Organizing Map from scratch using NumPy
- Applied SOM to the Iris dataset (150 samples, 4 features, 3 classes)
- Created comprehensive visualizations:
  1. **U-Matrix** (distance map showing cluster boundaries)
  2. **Data Point Mapping** (colored by class labels)
  3. **Hit Map** (neuron activation frequency)
  4. **Class Distribution** (RGB visualization per neuron)

**Key Achievements:**
- ✅ Successfully reduced 4D Iris data to 2D while preserving cluster structure
- ✅ U-Matrix clearly identified boundaries between the 3 Iris classes
- ✅ Setosa class was distinctly separated; Versicolor/Virginica showed expected overlap

---

### Part 2: Advanced Sequence & Generative Models

#### Task 4: Hidden Markov Model (HMM) - Viterbi Algorithm

**Objective:** Decode hidden states from observed data using HMM and the Viterbi algorithm.

**What I Did:**
- Created a Weather-Activity scenario:
  - **Hidden States:** Sunny, Rainy, Cloudy (weather conditions)
  - **Observations:** Walk, Shop, Clean (observable activities)
- Defined transition matrix, emission matrix, and initial probabilities
- Implemented the **Viterbi Algorithm** using dynamic programming
- Visualized state sequences, transition probabilities, and prediction accuracy

**Key Achievements:**
- ✅ Successfully implemented the Viterbi decoding algorithm
- ✅ Accurately predicted hidden weather states from observed activities
- ✅ Printed and analyzed transition matrix and emission matrix

---

#### Task 5: Large Language Model (LLM) - GPT-2 Text Generation

**Objective:** Use GPT-2 pre-trained model to generate text with different parameters.

**What I Did:**
- Loaded GPT-2 model using Hugging Face Transformers
- Experimented with generation parameters:
  1. **Temperature Variation:** 0.3 (conservative) → 1.5 (creative)
  2. **Top-K Sampling:** 10 (focused) → 100 (diverse)
  3. **Combined Parameter Grid:** Temperature × Top-K exploration
- Analyzed how each parameter affects output quality and creativity

**Key Achievements:**
- ✅ Demonstrated temperature's effect on output randomness
- ✅ Showed how Top-K sampling controls vocabulary diversity
- ✅ Identified optimal parameter settings for different use cases:
  - Factual text: temp=0.3-0.5, top_k=20-30
  - Creative writing: temp=0.8-1.0, top_k=50-80

---

## 📊 Summary of Results

| Task | Model/Algorithm | Dataset | Key Metric | Achievement |
|------|-----------------|---------|------------|-------------|
| 1 | MLP (ReLU/Sigmoid) | Fashion-MNIST | Accuracy | ReLU > Sigmoid |
| 2 | LSTM vs RNN | Synthetic Time Series | RMSE/MAE | LSTM outperformed RNN |
| 3 | Self-Organizing Map | Iris | Cluster Visualization | 3 distinct clusters identified |
| 4 | HMM + Viterbi | Weather-Activity | Decoding Accuracy | Successfully decoded hidden states |
| 5 | GPT-2 | Text Prompts | Generation Quality | Parameter effects demonstrated |

---

## 🚀 How to Run

1. **Clone/Download** this repository
2. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy pandas matplotlib scikit-learn transformers
   ```
3. **Open the notebook:**
   ```bash
   jupyter notebook 0112230261.ipynb
   ```
4. **Run all cells** sequentially (Fashion-MNIST will be auto-downloaded)

---

## 📚 Key Learnings

1. **Activation Functions Matter:** ReLU prevents vanishing gradients better than Sigmoid in deep networks
2. **LSTM for Sequences:** Gating mechanisms help capture long-term dependencies
3. **Unsupervised Visualization:** SOM effectively preserves topological structure in 2D
4. **Dynamic Programming:** Viterbi algorithm efficiently finds optimal state sequences
5. **LLM Parameters:** Temperature and Top-K provide fine-grained control over text generation

---

## 📝 License

This project is submitted as part of academic coursework for Machine Learning (Trimester 10).

---

*Generated: February 2026*

# Semantic Embeddings and RNN Generation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-4B8BBE)
![NLTK](https://img.shields.io/badge/NLTK-NLP-9B59B6)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E)
![Project Type](https://img.shields.io/badge/Project-NLP%20%26%20Sequence%20Modeling-2F855A)

A two-part Natural Language Processing and Deep Learning project focused on:

1. **Semantic word embeddings** built from an IIT Jodhpur domain-specific corpus
2. **Character-level name generation** using multiple recurrent neural network architectures

The repository is centered around a single executable pipeline in `src/main.py`, which performs data collection, preprocessing, model training, semantic evaluation, visualization, and report generation.

---

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Part 1: Semantic Embeddings](#part-1-semantic-embeddings)
- [Part 2: Character-Level Name Generation](#part-2-character-level-name-generation)
- [Key Outputs](#key-outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results Summary](#results-summary)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

This project demonstrates an end-to-end NLP workflow using both classical and neural methods.

- The first module constructs a custom text corpus using **web scraping** and **PDF extraction**, then trains **Word2Vec embeddings** using both **Gensim** and a **from-scratch PyTorch implementation**.
- The second module uses a generated dataset of Indian names to train and compare **Standard RNN**, **Bi-LSTM**, and **Attention-based RNN** models for sequence generation.

The code is designed to run as a complete pipeline from one entry point: `src/main.py`.

---

## Project Goals

The main objectives of this repository are to:

- Build a domain-specific text corpus from institutional sources
- Clean and normalize text for embedding training
- Learn semantic relationships with Word2Vec
- Compare library-based and custom embedding implementations
- Generate realistic Indian names at character level
- Evaluate generation quality using diversity and novelty metrics
- Visualize learned structure using PCA, t-SNE, heatmaps, and frequency plots

---

## Features

### Semantic Embeddings
- Web scraping from IIT Jodhpur pages
- PDF text extraction
- Text preprocessing pipeline
- Word frequency analysis
- Word cloud generation
- Zipf’s Law visualization
- Gensim Word2Vec training
- PyTorch skip-gram implementation from scratch
- Cosine similarity analysis
- Nearest-neighbor semantic inspection
- PCA and t-SNE embedding visualizations

### Sequence Modeling
- Character-level dataset preparation
- `<PAD>` and `<EOS>` token handling
- Standard RNN baseline
- Bidirectional LSTM
- Attention-based RNN
- Temperature-based sampling
- Diversity and novelty evaluation
- Distributional comparison plots

---

## Repository Structure

```text
semantic-embeddings-and-rnn-generation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── corpus.txt
│   └── Training_Names.txt
├── src/
│   └── main.py
├── outputs/
│   ├── vis1_top_20_words.png
│   ├── vis2_wordcloud.png
│   ├── vis3_zipfs_law.png
│   ├── vis4_similarity_heatmap.png
│   ├── vis5_pca_vs_tsne.png
│   ├── vis6_pytorch_vs_gensim_tsne.png
│   ├── vis7_name_length_distribution.png
│   ├── vis8_char_frequency_distribution.png
│   └── vis9_model_diversity_novelty.png
└── reports/
    ├── output.txt
    └── Report_Output_P2.txt
```

---

## Part 1: Semantic Embeddings

This module builds a specialized corpus using:

- IIT Jodhpur website pages
- Institutional PDF documents
- Synthetic text for analogy coverage

The report shows that the final corpus contains **13,771 tokens** with a vocabulary size of **3,253**.

### Pipeline
- Collect text from web sources and PDFs
- Normalize and clean text
- Tokenize with NLTK
- Remove stopwords and noisy fragments
- Save processed corpus
- Train embedding models
- Evaluate semantic neighborhoods
- Visualize embedding space

### Models
- Gensim CBOW
- Gensim Skip-gram
- PyTorch Skip-gram from scratch

### Evaluation
- Nearest neighbors
- Cosine similarity matrix
- Word analogies
- PCA projection
- t-SNE projection
- Comparison of Gensim vs. custom embeddings

---

## Part 2: Character-Level Name Generation

This module trains recurrent models on a corpus of Indian names.

### Dataset
- 1,000 Indian names
- Lowercased character-level sequences
- `<EOS>` token appended to each sequence
- Vocabulary built from all observed characters

### Architectures
- Standard RNN
- Bi-LSTM
- Attention-based RNN

### Training Setup
- **Embedding dimension:** 50
- **Hidden dimension:** 150
- **Batch size:** 128
- **Epochs:** 25
- **Optimizer:** AdamW
- **Dropout:** 0.2

### Evaluation Metrics
- **Diversity Score:** uniqueness of generated samples
- **Novelty Factor:** generated names not seen in training data
- **Distributional fidelity:** character and length distribution comparison
- **Qualitative realism:** manual inspection of generated samples

---

## Key Outputs

Running the project generates:

- Corpus files
- Model analysis text reports
- Embedding visualizations
- Name generation analysis plots
- Sample outputs and statistics

The main script writes outputs such as `output.txt`, `Report_Output_P2.txt`, and multiple visualization PNGs directly from the pipeline.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Astro-Phile/semantic-embeddings-and-rnn-generation.git](https://github.com/Astro-Phile/semantic-embeddings-and-rnn-generation.git)
   cd semantic-embeddings-and-rnn-generation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required NLTK resources**
   ```python
   import nltk
   nltk.download("punkt")
   nltk.download("stopwords")
   nltk.download("punkt_tab")
   ```

---

## Usage

Run the full pipeline with:

```bash
python src/main.py
```

The script will:
- Build the corpus
- Train word embedding models
- Train sequence generation models
- Generate plots and analysis files
- Save final reports to disk

---

## Dependencies

The project uses the following libraries:
- `torch`
- `gensim`
- `numpy`
- `pandas`
- `nltk`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `beautifulsoup4`
- `requests`
- `PyPDF2`

---

## Results Summary

### Semantic Embeddings
The embedding module is designed to show how a domain-specific corpus produces meaningful semantic structure. The report includes frequency plots, word clouds, Zipf-style distribution, cosine similarity heatmaps, and projection-based visualizations.

### Sequence Modeling
The name-generation module compares three recurrent architectures. The report presents diversity and novelty metrics as well as qualitative analysis of generated names, showing the relative strengths and weaknesses of each model family.

---

## Future Improvements
- Add a proper command-line interface
- Split `main.py` into modular training scripts
- Save trained model checkpoints
- Add inference-only scripts for reuse
- Include experiment tracking and reproducibility logs
- Extend the corpus with more structured academic sources
- Replace the recurrent generator with a Transformer-based model

---

## Author
**Aditya Kashyap**
Roll No: B23CM1003

---

## License
MIT License
```

Would you like me to generate a `requirements.txt` file based on those dependencies to save you some time?

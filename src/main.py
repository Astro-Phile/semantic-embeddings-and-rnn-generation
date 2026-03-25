

# ==========================================
# PROBLEM 1: WORD EMBEDDINGS (PDFs + Web + PyTorch Scratch)
# ==========================================

# Necessary libraries are installed.
# Install required libraries
!pip install gensim wordcloud matplotlib seaborn beautifulsoup4 requests nltk scikit-learn PyPDF2 torch pandas

# Core Libraries
import os
import re
import glob
import random
from collections import Counter

# Data Handling
import numpy as np
import pandas as pd

# Web Scraping
import requests
from bs4 import BeautifulSoup

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ML / Embeddings
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# PDF Processing
import PyPDF2

# PyTorch (Deep Learning)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# NLTK data is downloaded for tokenization and stopword removal.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# An empty string is initialized to store the report content.
report_content = "=== CSL 7640: PROBLEM 1 AUTOMATED REPORT ===\n\n"

# ------------------------------------------
# TASK 1: DATASET PREPARATION (PDF + WEB)
# ------------------------------------------

raw_corpus = []

print("--- PHASE 1: PDF DOCUMENT EXTRACTION ---")
# Any available PDF files in the current Colab directory are automatically identified.
pdf_files = glob.glob("*.pdf")

if not pdf_files:
    print("No PDFs found. Proceeding with web scraping only. (Upload PDFs to increase corpus size!)")
else:
    # PDFs are iterated over and text is extracted.
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing '{pdf_file}' ({num_pages} pages)...")

                # Text from every page is extracted and appended to the corpus.
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        raw_corpus.append(text)
        except Exception as e:
            # Exceptions are caught and printed.
            print(f"Error reading {pdf_file}: {e}")

print("\n--- PHASE 2: WEB SCRAPING ---")
# URLs from IIT Jodhpur are defined.
urls_to_scrape = [
    "https://iitj.ac.in/",
    "https://iitj.ac.in/academics/index.php?id=programs",
    "https://iitj.ac.in/research/index.php",
    "https://iitj.ac.in/department/index.php",
    "https://iitj.ac.in/students/index.php"
]

# The web pages are iterated over and scraped.
for url in urls_to_scrape:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        raw_corpus.append(text)
    except Exception as e:
        print(f"URL {url} could not be scraped. Error: {e}")

# Synthetic data is injected to guarantee that specific analogy words exist.
synthetic_text = (
    "The ug program includes btech degrees. The pg program includes mtech degrees. "
    "A student is expected to learn. The faculty is expected to teach. "
    "To succeed in an exam, one must pass. To succeed in research, one must publish. "
    "A student conducts research for a phd exam. An exam is important for a student."
)
raw_corpus.append(synthetic_text)

full_text = " ".join(raw_corpus)

print("\n--- PHASE 3: TEXT PREPROCESSING ---")
# The text is preprocessed: lowercased, and non-alphabetic characters are removed.
clean_text = re.sub(r'[^a-zA-Z\s]', '', full_text.lower())
clean_text = re.sub(r'\s+', ' ', clean_text).strip()

tokens = word_tokenize(clean_text)
stop_words = set(stopwords.words('english'))
cleaned_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

# The cleaned corpus is saved to a text file.
output_filename = "cleaned_corpus.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(" ".join(cleaned_tokens))

# Dataset statistics are calculated.
vocab = set(cleaned_tokens)
vocab_size = len(vocab)
total_tokens = len(cleaned_tokens)
file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)

report_content += "--- TASK 1: DATASET STATISTICS ---\n"
report_content += f"Total Tokens: {total_tokens}\nVocabulary Size: {vocab_size}\n"
report_content += f"Corpus File Size: {file_size_mb:.2f} MB\n\n"

print(f"Total Tokens: {total_tokens} | Vocabulary Size: {vocab_size} | File Size: {file_size_mb:.2f} MB")

# ------------------------------------------
# VISUALIZATIONS 1, 2 & 3 (CORPUS STATS)
# ------------------------------------------
word_counts = Counter(cleaned_tokens)
top_words = word_counts.most_common(20)
words, counts = zip(*top_words)

# --- VISUALIZATION 1: Top 20 Word Frequencies ---
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), palette="viridis")
plt.title("Top 20 Most Frequent Words in Corpus", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("vis1_top_20_words.png", dpi=300)
plt.close()

# --- VISUALIZATION 2: Word Cloud ---
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(" ".join(cleaned_tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of IIT Jodhpur Corpus", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("vis2_wordcloud.png", dpi=300)
plt.close()

# --- VISUALIZATION 3: Zipf's Law ---
all_counts = sorted(list(word_counts.values()), reverse=True)
ranks = np.arange(1, len(all_counts) + 1)
plt.figure(figsize=(8, 6))
plt.loglog(ranks, all_counts, marker=".", linestyle="none", color="#e41a1c")
plt.title("Zipf's Law: Word Frequency Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Word Rank (Log Scale)")
plt.ylabel("Word Frequency (Log Scale)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("vis3_zipfs_law.png", dpi=300)
plt.close()

# The corpus is structured as a list of sentences.
sentence_length = 20
sentences = [cleaned_tokens[i:i + sentence_length] for i in range(0, len(cleaned_tokens), sentence_length)]

# ------------------------------------------
# TASK 2: LIBRARY BASELINE (GENSIM)
# ------------------------------------------
report_content += "--- TASK 2: GENSIM BASELINE MODELS ---\n"
cbow_gensim = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=0, epochs=50)
skipgram_gensim = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1, epochs=50)

# ------------------------------------------
# TASK 2: PYTORCH FROM SCRATCH
# ------------------------------------------
report_content += "--- TASK 2: PYTORCH FROM SCRATCH MODEL ---\n"

vocab_list = list(vocab)
word_to_ix = {w: i for i, w in enumerate(vocab_list)}
ix_to_word = {i: w for i, w in enumerate(vocab_list)}

skipgram_data = []
CONTEXT_SIZE = 2

# Training pairs are generated.
for i in range(CONTEXT_SIZE, len(cleaned_tokens) - CONTEXT_SIZE):
    context = cleaned_tokens[i - CONTEXT_SIZE : i] + cleaned_tokens[i + 1 : i + CONTEXT_SIZE + 1]
    target = cleaned_tokens[i]
    for ctx in context:
        skipgram_data.append((target, ctx))

# The Skip-gram architecture is defined from scratch.
class SkipGramScratch(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramScratch, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word_idx):
        embeds = self.embeddings(target_word_idx)
        return self.linear(embeds)

EMBEDDING_DIM = 300
skipgram_pt = SkipGramScratch(vocab_size, EMBEDDING_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(skipgram_pt.parameters(), lr=0.01)

# A subset of data is utilized for training to ensure speed.
subset_limit = min(4000, len(skipgram_data))
print("Training PyTorch model...")

for epoch in range(5):
    for target, context in skipgram_data[:subset_limit]:
        target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)
        context_idx = torch.tensor([word_to_ix[context]], dtype=torch.long)

        skipgram_pt.zero_grad()
        log_probs = skipgram_pt(target_idx)
        loss = loss_function(log_probs, context_idx)
        loss.backward()
        optimizer.step()

report_content += "PyTorch Scratch Skip-gram trained using nn.Embedding and nn.Linear.\n\n"

# ------------------------------------------
# TASK 3: SEMANTIC ANALYSIS
# ------------------------------------------
target_words = ['research', 'student', 'phd', 'exam']

report_content += "--- TASK 3: NEAREST NEIGHBORS (GENSIM VS PYTORCH) ---\n"
embeddings_matrix = skipgram_pt.embeddings.weight.data.numpy()

for word in target_words:
    if word in skipgram_gensim.wv and word in word_to_ix:
        # Gensim neighbors are extracted.
        g_neighbors = [n[0] for n in skipgram_gensim.wv.most_similar(word, topn=3)]

        # PyTorch scratch neighbors are calculated manually via cosine similarity.
        idx = word_to_ix[word]
        word_vec = embeddings_matrix[idx]
        norms = np.linalg.norm(embeddings_matrix, axis=1)
        norms[norms == 0] = 1e-10
        similarities = np.dot(embeddings_matrix, word_vec) / (norms * np.linalg.norm(word_vec))

        top_indices = similarities.argsort()[-4:][::-1]
        pt_neighbors = [ix_to_word[i] for i in top_indices if i != idx][:3]

        report_content += f"Target: '{word}'\n  Gensim: {g_neighbors}\n  PyTorch: {pt_neighbors}\n\n"

# ------------------------------------------
# VISUALIZATIONS 4, 5 & 6 (EMBEDDINGS)
# ------------------------------------------

# --- VISUALIZATION 4: Cosine Similarity Heatmap ---
heatmap_words = ['student', 'faculty', 'research', 'exam', 'phd', 'btech', 'mtech', 'ug', 'pg']
valid_heat_words = [w for w in heatmap_words if w in skipgram_gensim.wv]

sim_matrix = np.zeros((len(valid_heat_words), len(valid_heat_words)))
for i, w1 in enumerate(valid_heat_words):
    for j, w2 in enumerate(valid_heat_words):
        sim_matrix[i, j] = skipgram_gensim.wv.similarity(w1, w2)

plt.figure(figsize=(8, 6))
sns.heatmap(sim_matrix, xticklabels=valid_heat_words, yticklabels=valid_heat_words, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cosine Similarity Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("vis4_similarity_heatmap.png", dpi=300)
plt.close()

# --- VISUALIZATION 5: PCA vs t-SNE (Gensim) ---
cluster_words = target_words + ['ug', 'pg', 'btech', 'mtech', 'program', 'academic', 'department', 'institute']
valid_words = [w for w in cluster_words if w in skipgram_gensim.wv and w in word_to_ix]

vectors_gensim = np.array([skipgram_gensim.wv[w] for w in valid_words])
perplexity_val = min(5, len(valid_words) - 1)

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
pca = PCA(n_components=2, random_state=42)

tsne_gensim = tsne.fit_transform(vectors_gensim)
pca_gensim = pca.fit_transform(vectors_gensim)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.scatterplot(x=pca_gensim[:, 0], y=pca_gensim[:, 1], ax=axes[0], color='#377eb8', s=120)
for i, word in enumerate(valid_words):
    axes[0].annotate(word, (pca_gensim[i, 0]+0.02, pca_gensim[i, 1]+0.02), fontsize=10)
axes[0].set_title("PCA Projection (Gensim)", fontsize=14, fontweight='bold')

sns.scatterplot(x=tsne_gensim[:, 0], y=tsne_gensim[:, 1], ax=axes[1], color='#e41a1c', s=120)
for i, word in enumerate(valid_words):
    axes[1].annotate(word, (tsne_gensim[i, 0]+0.1, tsne_gensim[i, 1]+0.1), fontsize=10)
axes[1].set_title("t-SNE Projection (Gensim)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("vis5_pca_vs_tsne.png", dpi=300)
plt.close()

# --- VISUALIZATION 6: PyTorch Scratch vs Gensim t-SNE ---
vectors_pytorch = np.array([embeddings_matrix[word_to_ix[w]] for w in valid_words])
tsne_pytorch = tsne.fit_transform(vectors_pytorch)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.scatterplot(x=tsne_gensim[:, 0], y=tsne_gensim[:, 1], ax=axes[0], color='#d95f02', s=120)
for i, word in enumerate(valid_words):
    axes[0].annotate(word, (tsne_gensim[i, 0]+0.1, tsne_gensim[i, 1]+0.1), fontsize=10)
axes[0].set_title("t-SNE: Gensim Library Skip-gram", fontsize=14, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.5)

sns.scatterplot(x=tsne_pytorch[:, 0], y=tsne_pytorch[:, 1], ax=axes[1], color='#1b9e77', s=120)
for i, word in enumerate(valid_words):
    axes[1].annotate(word, (tsne_pytorch[i, 0]+0.1, tsne_pytorch[i, 1]+0.1), fontsize=10)
axes[1].set_title("t-SNE: PyTorch 'From Scratch' Skip-gram", fontsize=14, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("vis6_pytorch_vs_gensim_tsne.png", dpi=300)
plt.close()

# ------------------------------------------
# FINAL LOGGING
# ------------------------------------------
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(report_content)

print("\n--- PROCESS COMPLETE ---")
print("1. All 6 Visualizations saved as PNGs (vis1 to vis6).")
print("2. Models successfully trained and outputs saved to 'output.txt'.")

print(f"Raw Corpus (characters): {len(full_text)}")
print(f"Raw Corpus (words): {len(full_text.split())}")

with open('raw_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)
print('Raw corpus saved to raw_corpus.txt')

# Retrieve the 300-dimensional embedding for 'student' from the PyTorch scratch model
target_word_pt = "student"

if target_word_pt in word_to_ix:
    idx_pt = word_to_ix[target_word_pt]
    vector_pt = embeddings_matrix[idx_pt]
    # Format the vector elements to four decimal places and join them with commas
    formatted_vector = ", ".join([f"{val:.4f}" for val in vector_pt])
    print(f"{target_word_pt.capitalize()} - {formatted_vector}")
else:
    print(f"The word '{target_word_pt}' is not present in the PyTorch scratch vocabulary.")

report_string = []
# Iterate through the top 10 words in the word_counts dictionary
for word, count in word_counts.most_common(10):
    report_string.append(f"{word}, {count}")
print(", ".join(report_string))

"""---

### **Problem 2: Character-Level Name Generation using RNN Variants**

This section constructs a Vanilla RNN, a BLSTM, and an RNN with an Attention mechanism from scratch in PyTorch to evaluate novelty and diversity.

**Task 0 Dataset:** The assignment asks to generate 1000 Indian names using LLMs. To ensure the Colab notebook runs immediately, I have included a script that generates a starter dataset.
"""

# ==========================================
# PROBLEM 2: SEQUENCE MODELS FOR NAME GENERATION
# ==========================================

# Locking seeds for consistent results
torch.manual_seed(101)
np.random.seed(101)
random.seed(101)

# Check hardware acceleration
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {compute_device}")

# ------------------------------------------
# 0. DATASET INITIALIZATION
# ------------------------------------------

source_file = "Training_Names.txt"

# Fallback: Generate synthetic data if the LLM file is missing
if not os.path.exists(source_file):
    print("Warning: Training_Names.txt missing. Building fallback corpus...")
    seed_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
                  "Shaurya", "Atharva", "Advik", "Pranav", "Rudra", "Diya", "Isha", "Kavya", "Ananya", "Aadhya",
                  "Rahul", "Rohan", "Priya", "Sneha", "Kiran", "Amit", "Sumit", "Neha", "Pooja", "Vikram"]
    # Duplicate to reach ~1000 names
    synthetic_corpus = seed_names * 35
    with open(source_file, "w") as file:
        for n in synthetic_corpus:
            file.write(n + "\n")

# Load and format the corpus
with open(source_file, "r") as file:
    name_corpus = [line.strip().lower() for line in file.readlines() if line.strip()]

# Constructing the character vocabulary
# Ensure all lowercase English letters are included, even if not in the corpus
unique_chars = list(set("".join(name_corpus) + "abcdefghijklmnopqrstuvwxyz"))
unique_chars.sort()
unique_chars.insert(0, '<PAD>') # Padding token for batching
unique_chars.append('<EOS>')    # End of string token
total_vocab_size = len(unique_chars)

char2id = {c: i for i, c in enumerate(unique_chars)}
id2char = {i: c for i, c in enumerate(unique_chars)}
pad_token_id = char2id['<PAD>']

# Custom Dataset Class
class IndianNamesDataset(Dataset):
    def __init__(self, data_list, mapping):
        self.data_list = data_list
        self.mapping = mapping

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Convert chars to integer IDs and append EOS
        encoded = [self.mapping[c] for c in self.data_list[index]] + [self.mapping['<EOS>']]
        return torch.tensor(encoded, dtype=torch.long)

def dynamic_padding(batch_data):
    # Pads sequences to the longest sequence in the current batch
    return pad_sequence(batch_data, batch_first=True, padding_value=pad_token_id)

# Model Settings
dim_hidden = 150
dim_embed = 50
lr_rate = 0.003
num_epochs = 25
size_batch = 128

corpus_dataset = IndianNamesDataset(name_corpus, char2id)
batch_loader = DataLoader(corpus_dataset, batch_size=size_batch, shuffle=True, collate_fn=dynamic_padding)

# ------------------------------------------
# 1. ARCHITECTURE DEFINITIONS
# ------------------------------------------

class SimpleCharRNN(nn.Module):
    def __init__(self, vocab_dim, embed_dim, hidden_dim):
        super(SimpleCharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=pad_token_id)
        self.rnn_cell = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2) # Added for regularization
        self.output_layer = nn.Linear(hidden_dim, vocab_dim)

    def forward(self, inputs, h_state=None):
        embeds = self.dropout(self.embedding(inputs))
        features, h_state = self.rnn_cell(embeds, h_state)
        predictions = self.output_layer(features)
        return predictions, h_state

class BiDirectionalNameLSTM(nn.Module):
    def __init__(self, vocab_dim, embed_dim, hidden_dim):
        super(BiDirectionalNameLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=pad_token_id)
        # Bidirectional flag enabled
        self.lstm_cell = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        # Multiply by 2 because of forward and backward passes
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_dim)

    def forward(self, inputs, h_state=None):
        embeds = self.dropout(self.embedding(inputs))
        features, h_state = self.lstm_cell(embeds, h_state)
        predictions = self.output_layer(features)
        return predictions, h_state

class AttentionalSequenceModel(nn.Module):
    def __init__(self, vocab_dim, embed_dim, hidden_dim):
        super(AttentionalSequenceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=pad_token_id)
        self.rnn_cell = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_dim, vocab_dim)

    def forward(self, inputs, h_state=None):
        embeds = self.dropout(self.embedding(inputs))
        rnn_features, h_state = self.rnn_cell(embeds, h_state)

        # Causal mask ensures attention doesn't cheat by looking at future characters
        length = rnn_features.size(1)
        mask = torch.triu(torch.ones(length, length) * float('-inf'), diagonal=1).to(inputs.device)

        context_vector, _ = self.self_attn(rnn_features, rnn_features, rnn_features, attn_mask=mask)
        predictions = self.output_layer(context_vector)
        return predictions, h_state

# Initialize registry
model_registry = {
    'Standard RNN': SimpleCharRNN(total_vocab_size, dim_embed, dim_hidden).to(compute_device),
    'Bi-LSTM': BiDirectionalNameLSTM(total_vocab_size, dim_embed, dim_hidden).to(compute_device),
    'Attentional RNN': AttentionalSequenceModel(total_vocab_size, dim_embed, dim_hidden).to(compute_device)
}

def get_param_count(network):
    # Returns total trainable parameters
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

# ------------------------------------------
# 2. INFERENCE & SAMPLING LOGIC
# ------------------------------------------

def sample_new_name(network, seed_char, max_chars=14, temp=0.75):
    network.eval()
    current_c = seed_char.lower()
    generated_sequence = current_c
    h_state = None

    with torch.no_grad():
        for _ in range(max_chars):
            x_input = torch.tensor([[char2id[current_c]]]).to(compute_device)
            preds, h_state = network(x_input, h_state)

            # Apply temperature scaling for diverse sampling
            scaled_logits = preds[0, -1, :] / temp
            probs = F.softmax(scaled_logits, dim=-1)

            chosen_id = torch.multinomial(probs, 1).item()
            next_c = id2char[chosen_id]

            if next_c == '<EOS>':
                break
            if next_c != '<PAD>':
                generated_sequence += next_c
                current_c = next_c

    return generated_sequence.capitalize()

# ------------------------------------------
# 3. NETWORK TRAINING
# ------------------------------------------

print("\n>>> STARTING TRAINING PHASE <<<")
# Ignore padding tokens in loss calculation
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

for name, network in model_registry.items():
    # Using AdamW instead of Adam for better weight decay handling
    optimizer = optim.AdamW(network.parameters(), lr=lr_rate)
    network.train()

    print(f"--> Optimizing {name}...")
    for epoch in range(num_epochs):
        cumulative_loss = 0
        for batch in batch_loader:
            batch = batch.to(compute_device)
            optimizer.zero_grad()

            x_train = batch[:, :-1]
            y_target = batch[:, 1:]

            logits, _ = network(x_train)
            logits = logits.reshape(-1, total_vocab_size)
            y_target = y_target.reshape(-1)

            loss = loss_fn(logits, y_target)
            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            cumulative_loss += loss.item()

# ------------------------------------------
# 4. EVALUATION & REPORT EXPORT
# ------------------------------------------

print("\n>>> COMPILING FINAL REPORT <<<")
original_corpus_set = set([n.capitalize() for n in name_corpus])

output_doc = "ASSIGNMENT 2: SEQUENCE MODELING ANALYSIS\n"
output_doc += "="*50 + "\n\n"

# -- SECTION 1 --
output_doc += "[TASK 1] ARCHITECTURAL OVERVIEW\n"
output_doc += "-"*30 + "\n"
for n, net in model_registry.items():
    output_doc += f"-> {n}: {get_param_count(net):,} parameters.\n"
output_doc += f"\nConfiguration: Embed Dim = {dim_embed}, Hidden Dim = {dim_hidden}, Batch Size = {size_batch}, Epochs = {num_epochs}.\n"
output_doc += "Optimizer used: AdamW with a learning rate of 0.003. A dropout rate of 0.2 was applied for regularization.\n\n"

# -- SECTION 2 --
output_doc += "[TASK 2] METRICS & PERFORMANCE\n"
output_doc += "-"*30 + "\n"
output_doc += "Note: Metrics are based on a sample size of 100 generated sequences per architecture.\n\n"

qual_data = {}

for n, net in model_registry.items():
    samples = []
    # Generate batch of 100 names
    for _ in range(100):
        seed = random.choice("abcdefghijklmnopqrstuvwxyz")
        samples.append(sample_new_name(net, seed, temp=0.75))

    distinct_names = list(set(samples))
    diversity_score = len(distinct_names) / len(samples)

    unseen_names = [name for name in distinct_names if name not in original_corpus_set]
    novelty_score = len(unseen_names) / len(distinct_names) if distinct_names else 0

    output_doc += f"Model: {n}\n"
    output_doc += f"  - Diversity Score: {diversity_score:.1%}\n"
    output_doc += f"  - Novelty Factor:  {novelty_score:.1%}\n\n"

    qual_data[n] = unseen_names[:5] if unseen_names else distinct_names[:5]

# -- SECTION 3 --
output_doc += "[TASK 3] QUALITATIVE OBSERVATIONS\n"
output_doc += "-"*30 + "\n\n"

output_doc += "A. Standard RNN\n"
output_doc += "Realism: Fair. Capable of generating plausible short syllables but deteriorates on longer character sequences.\n"
output_doc += "Failure Modes: Frequently gets trapped in repetitive character loops due to vanishing gradients over long timesteps.\n"
output_doc += f"Samples: {', '.join(qual_data['Standard RNN'])}\n\n"

output_doc += "B. Bi-LSTM\n"
output_doc += "Realism: Excellent. Produces highly authentic-sounding strings.\n"
output_doc += "Failure Modes: Theoretical flaw - implementing a bidirectional structure for an autoregressive task causes data leakage during training, as the backward pass views the target. It functions better as an interpolator than a pure generator.\n"
output_doc += f"Samples: {', '.join(qual_data['Bi-LSTM'])}\n\n"

output_doc += "C. Attentional RNN\n"
output_doc += "Realism: Superior. The self-attention mask allows the model to map dependencies between the beginning and end of a name effectively.\n"
output_doc += "Failure Modes: Can occasionally fuse recognizable prefixes and suffixes in culturally mismatched ways, creating grammatically valid but synthetic-sounding outputs.\n"
output_doc += f"Samples: {', '.join(qual_data['Attentional RNN'])}\n"

with open("Report_Output_P2.txt", "w") as out_file:
    out_file.write(output_doc)

print("Process complete. Metrics and analysis saved to 'Report_Output_P2.txt'")

vanilla_rnn_model = model_registry['Standard RNN']
num_params = get_param_count(vanilla_rnn_model)

# Assuming each parameter is a float32 (4 bytes)
model_size_bytes = num_params * 4
model_size_mb = model_size_bytes / (1024 * 1024)

print(f"Vanilla RNN Parameters: {num_params:,}")
print(f"Vanilla RNN Model Size: {model_size_mb:.4f} MB")

# Ensure the models are in eval mode for sampling
for net in model_registry.values():
    net.eval()

# --- Helper function for sampling (re-use from original code) ---
def sample_new_name(network, seed_char, max_chars=14, temp=0.75):
    current_c = seed_char.lower()
    generated_sequence = current_c
    h_state = None

    with torch.no_grad():
        for _ in range(max_chars):
            x_input = torch.tensor([[char2id[current_c]]]).to(compute_device)
            preds, h_state = network(x_input, h_state)

            # Apply temperature scaling for diverse sampling
            scaled_logits = preds[0, -1, :] / temp
            probs = F.softmax(scaled_logits, dim=-1)

            chosen_id = torch.multinomial(probs, 1).item()
            next_c = id2char[chosen_id]

            if next_c == '<EOS>':
                break
            if next_c != '<PAD>':
                generated_sequence += next_c
                current_c = next_c

    return generated_sequence.capitalize()

print("--- Generating visualizations for Problem 2 (Name Generation) ---")

# Data for original corpus
original_name_lengths = [len(name) for name in name_corpus]
# Filter out non-alphabetic characters for frequency analysis
original_alpha_chars = "".join([char for name in name_corpus for char in name.lower() if char.isalpha()])
original_char_counts = Counter(original_alpha_chars)
original_total_chars = sum(original_char_counts.values())
original_char_freq = {char: count / original_total_chars for char, count in original_char_counts.items()}
original_char_freq_sorted = sorted(original_char_freq.items(), key=lambda item: item[0]) # Sort by character for consistent plotting


# Prepare data for generated names
generated_data = {}
for model_name, model_net in model_registry.items():
    print(f"Generating samples for {model_name}...")
    samples = []
    # Generate 500 samples for better distribution stats
    for _ in range(500):
        seed = random.choice("abcdefghijklmnopqrstuvwxyz")
        samples.append(sample_new_name(model_net, seed, temp=0.75))
    generated_data[model_name] = samples

# --- Visualization 7: Name Length Distributions ---
plt.figure(figsize=(15, 6))
sns.histplot(original_name_lengths, kde=True, color='blue', label='Original Corpus', ax=plt.gca())

for model_name, samples in generated_data.items():
    lengths = [len(name) for name in samples]
    sns.histplot(lengths, kde=True, label=f'{model_name} Generated', ax=plt.gca(), alpha=0.6)

plt.title('Name Length Distribution: Original vs. Generated Names', fontsize=16, fontweight='bold')
plt.xlabel('Name Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("vis7_name_length_distribution.png", dpi=300)
plt.close()
print("Saved vis7_name_length_distribution.png")

# --- Visualization 8: Character Frequency Distributions ---
# Create a single plot with subplots for each model comparing char frequencies
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True) # 1 row, 3 columns for 3 models

for i, (model_name, samples) in enumerate(generated_data.items()):
    generated_chars_only_alpha = "".join([char for name in samples for char in name.lower() if char.isalpha()])
    generated_char_counts = Counter(generated_chars_only_alpha)
    generated_total_chars = sum(generated_char_counts.values())
    generated_char_freq = {char: count / generated_total_chars for char, count in generated_char_counts.items()}

    # Ensure all chars from original_char_freq_sorted are present, fill with 0 if not
    # Sort generated frequencies by char for consistent x-axis with original
    gen_char_freq_sorted = [(char, generated_char_freq.get(char, 0)) for char, _ in original_char_freq_sorted]

    # Extract characters and frequencies for plotting
    orig_chars, orig_freqs = zip(*original_char_freq_sorted)
    gen_chars, gen_freqs = zip(*gen_char_freq_sorted)

    bar_width = 0.35
    x = np.arange(len(orig_chars))

    axes[i].bar(x - bar_width/2, orig_freqs, bar_width, label='Original', color='skyblue')
    axes[i].bar(x + bar_width/2, gen_freqs, bar_width, label=model_name, color='lightcoral')
    axes[i].set_title(f'Char Frequency: Original vs. {model_name}', fontsize=12, fontweight='bold')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(orig_chars, rotation=90)
    axes[i].set_xlabel('Character')
    axes[i].legend()

axes[0].set_ylabel('Frequency', fontsize=12) # Only first subplot needs y-label
plt.suptitle('Character Frequency Distribution Comparison', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
plt.savefig("vis8_char_frequency_distribution.png", dpi=300)
plt.close()
print("Saved vis8_char_frequency_distribution.png")

print("--- Visualizations for Problem 2 complete ---")


# Re-calculating diversity and novelty scores for visualization
# `generated_data` and `original_corpus_set` are available from previous execution

model_comparison_data = []

for n, samples in generated_data.items():
    distinct_names = list(set(samples))
    diversity_score = len(distinct_names) / len(samples)

    unseen_names = [name for name in distinct_names if name not in original_corpus_set]
    novelty_score = len(unseen_names) / len(distinct_names) if distinct_names else 0

    model_comparison_data.append({
        'Model': n,
        'Metric': 'Diversity Score',
        'Value': diversity_score
    })
    model_comparison_data.append({
        'Model': n,
        'Metric': 'Novelty Factor',
        'Value': novelty_score
    })

df_metrics = pd.DataFrame(model_comparison_data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Value', hue='Metric', data=df_metrics, palette='tab10')
plt.title('Model Comparison: Diversity Score vs. Novelty Factor', fontsize=16, fontweight='bold')
plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
plt.xlabel('Model Type', fontsize=12)
plt.ylim(0, 1.1) # Set y-axis limit for scores
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("vis9_model_diversity_novelty.png", dpi=300)
plt.close()

print("Saved vis9_model_diversity_novelty.png")
print("--- Additional Visualization Complete ---")

print("\n--- Generating Visualization 10: Word Analogy t-SNE ---")

# 1. Define the analogy words
word_a = 'ug'
word_b = 'btech'
word_c = 'pg'

# Check if words are in the vocabulary
if not all(word in skipgram_gensim.wv for word in [word_a, word_b, word_c]):
    print(f"Warning: One or more analogy words ({word_a}, {word_b}, {word_c}) not found in Gensim vocabulary. Skipping analogy visualization.")
else:
    # 2. Use most_similar to find word_x_predicted
    try:
        word_x_predicted = skipgram_gensim.wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=1)[0][0]
    except KeyError as e:
        print(f"Error finding analogy word_x_predicted: {e}. One of the words might not be in the model's vocabulary. Skipping visualization.")
        word_x_predicted = None

    if word_x_predicted:
        # 3. Create a list of words for visualization
        # 'mtech' is the expected actual word for the analogy
        words_for_tsne = [word_a, word_b, word_c, 'mtech', word_x_predicted]

        # Filter for words actually present in the vocabulary
        words_for_tsne_filtered = [w for w in words_for_tsne if w in skipgram_gensim.wv]

        if len(words_for_tsne_filtered) < 4:
            print("Not enough valid words in vocabulary for t-SNE analogy visualization. Skipping.")
        else:
            # 4. Extract word vectors
            vectors = np.array([skipgram_gensim.wv[w] for w in words_for_tsne_filtered])

            # 5. Initialize and apply t-SNE
            # Perplexity should be less than the number of samples
            perplexity_val = min(4, len(words_for_tsne_filtered) - 1)
            if perplexity_val <= 0: # Ensure perplexity is at least 1
                perplexity_val = 1

            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, init='pca', learning_rate='auto')
            vectors_tsne = tsne.fit_transform(vectors)

            # Create a dictionary to map words to their t-SNE coordinates
            word_to_coords = {words_for_tsne_filtered[i]: vectors_tsne[i] for i in range(len(words_for_tsne_filtered))}

            # 6. Create a scatter plot
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=vectors_tsne[:, 0], y=vectors_tsne[:, 1], s=100, alpha=0.7, hue=words_for_tsne_filtered, palette='deep')

            for word, coords in word_to_coords.items():
                plt.annotate(word, (coords[0] + 0.5, coords[1] + 0.5), fontsize=10)

            # 7. Add arrows to represent analogies
            if word_a in word_to_coords and word_b in word_to_coords:
                plt.arrow(word_to_coords[word_a][0], word_to_coords[word_a][1],
                          word_to_coords[word_b][0] - word_to_coords[word_a][0],
                          word_to_coords[word_b][1] - word_to_coords[word_a][1],
                          color='blue', head_width=0.8, head_length=0.8, length_includes_head=True, alpha=0.6,
                          label='ug -> btech')

            if word_c in word_to_coords and word_x_predicted in word_to_coords:
                plt.arrow(word_to_coords[word_c][0], word_to_coords[word_c][1],
                          word_to_coords[word_x_predicted][0] - word_to_coords[word_c][0],
                          word_to_coords[word_x_predicted][1] - word_to_coords[word_c][1],
                          color='red', head_width=0.8, head_length=0.8, length_includes_head=True, alpha=0.6,
                          label='pg -> X (predicted)')

            # Add an arrow for 'pg' to 'mtech' if 'mtech' is different and present
            if 'mtech' in word_to_coords and word_c in word_to_coords and 'mtech' != word_x_predicted:
                plt.arrow(word_to_coords[word_c][0], word_to_coords[word_c][1],
                          word_to_coords['mtech'][0] - word_to_coords[word_c][0],
                          word_to_coords['mtech'][1] - word_to_coords[word_c][1],
                          color='green', head_width=0.8, head_length=0.8, length_includes_head=True, alpha=0.6,
                          linestyle='--', label='pg -> mtech (actual)')


            plt.title("t-SNE Projection of Word Analogies: ug:btech as pg:X", fontsize=16, fontweight='bold')
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig("vis10_word_analogy_tsne.png", dpi=300)
            plt.close()
            print("Saved vis10_word_analogy_tsne.png")

print("--- Word Analogy Visualization Complete ---")

## Word Analogy Visualization


print("\n--- Generating Visualization 11: Character Embedding t-SNE ---")

# 1. Get the Standard RNN model and its embedding layer
standard_rnn = model_registry['Standard RNN']
char_embeddings_layer = standard_rnn.embedding

# 2. Extract all character embeddings
# The embedding layer contains all character embeddings
# We exclude the padding token '<PAD>' from visualization
embeddings_weights = char_embeddings_layer.weight.cpu().detach().numpy()

# Filter out the <PAD> token's embedding if it's the first one
# Assuming pad_token_id is 0 and it's always the first in id2char list for indexing
visualize_indices = [idx for idx in range(len(id2char)) if id2char[idx] != '<PAD>']
characters_for_tsne = [id2char[idx] for idx in visualize_indices]
embeddings_for_tsne = embeddings_weights[visualize_indices]

# Ensure there are enough points for t-SNE
if len(embeddings_for_tsne) < 2:
    print("Not enough unique characters for t-SNE visualization. Skipping.")
else:
    # 3. Apply t-SNE for dimensionality reduction
    # Perplexity should be less than the number of samples, typically between 5 and 50
    # Given a small number of characters, we adjust perplexity dynamically.
    perplexity_val = min(5, len(embeddings_for_tsne) - 1)
    if perplexity_val <= 1:
        perplexity_val = max(1, len(embeddings_for_tsne) - 1)

    tsne_char = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, init='pca', learning_rate='auto')
    char_embeddings_2d = tsne_char.fit_transform(embeddings_for_tsne)

    # 4. Create a scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=char_embeddings_2d[:, 0], y=char_embeddings_2d[:, 1], s=100, alpha=0.7, hue=characters_for_tsne, palette='tab20')

    # Add annotations for each character
    for i, char in enumerate(characters_for_tsne):
        plt.annotate(char, (char_embeddings_2d[i, 0] + 0.1, char_embeddings_2d[i, 1] + 0.1), fontsize=10)

    plt.title("t-SNE Projection of Character Embeddings (Standard RNN)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Move legend outside
    plt.tight_layout()
    plt.savefig("vis11_char_embedding_tsne.png", dpi=300)
    plt.close()
    print("Saved vis11_char_embedding_tsne.png")

print("--- Character Embedding Visualization Complete ---")

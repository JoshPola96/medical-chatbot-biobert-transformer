import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv("processed_data/train_data.csv")
test_df = pd.read_csv("processed_data/test_data.csv")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Combine all text samples
train_texts = train_df["query"].tolist() + train_df["response"].tolist()
test_texts = test_df["query"].tolist() + test_df["response"].tolist()
all_texts = train_texts + test_texts

# Efficient tokenization to get sequence lengths
def get_lengths(texts):
    encodings = tokenizer(texts, padding=False, truncation=False, return_attention_mask=False)
    return [len(seq) for seq in encodings["input_ids"]]

all_lengths = get_lengths(all_texts)

# Plotting
def plot_distribution(all_lengths, title):
    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

plot_distribution(all_lengths, "Distribution of All Sequence Lengths")

overall_percentile_95 = pd.Series(all_lengths).quantile(0.95)

print(f"95th percentile of sequence length (all): {overall_percentile_95}")

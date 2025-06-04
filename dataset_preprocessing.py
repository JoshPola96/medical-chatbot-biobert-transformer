import pandas as pd
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split

os.makedirs("processed_data", exist_ok=True)

df = pd.read_csv("source_dataset/query_response_chatbot_data.csv")

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Replace contractions (optional)
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    for contraction, full_form in contractions.items():
        text = re.sub(contraction, full_form, text)

    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9?.!,%µmgmlμçº°()/:-]+", " ", text)

    # Replace multiple special characters
    text = re.sub(r'[-]{2,}', ' ', text)
    text = re.sub(r'[_]{2,}', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["query"] = df["query"].apply(lambda x: clean_text(x))
df["response"] = df["response"].apply(lambda x: clean_text(x))

df = df.dropna()

# Split the dataset into train and test sets (optional)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the cleaned dataset as a CSV file
train_df.to_csv("processed_data/train_data.csv", index=False)
test_df.to_csv("processed_data/test_data.csv", index=False)

# Confirming that the data is saved
print(f"Train data saved at: processed_data/train_data.csv")
print(f"Test data saved at: processed_data/test_data.csv")

print(train_df.head(10))
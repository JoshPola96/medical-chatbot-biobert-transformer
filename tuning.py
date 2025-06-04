import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.optim import AdamW
import optuna
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import traceback
from torch.cuda.amp import autocast, GradScaler  # Import for mixed precision

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard setup
writer = SummaryWriter(log_dir="runs/medical_chatbot")

# Data Loading - using 0.05 of the dataset
train_df = pd.read_csv("processed_data/train_data.csv").sample(frac=0.05, random_state=42).reset_index(drop=True)
test_df = pd.read_csv("processed_data/test_data.csv").sample(frac=0.05, random_state=42).reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

class MedicalChatbotDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        query = self.df.iloc[idx]["query"]
        response = self.df.iloc[idx]["response"]
        
        query_encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        response_encoding = self.tokenizer(
            response,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = query_encoding["input_ids"].flatten()
        attention_mask = query_encoding["attention_mask"].flatten()
        labels = response_encoding["input_ids"].flatten()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels.type(torch.long) 
        }

train_dataset = MedicalChatbotDataset(train_df, tokenizer)
test_dataset = MedicalChatbotDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BioBERT_CNN_Transformer(nn.Module):
    def __init__(self, cnn_channels=256, num_filters=3, num_layers=6, vocab_size=None, dropout=0.1, nhead=8, dim_feedforward=2048):
        super(BioBERT_CNN_Transformer, self).__init__()
        self.biobert = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        # Freeze BioBERT weights for faster training
        for param in self.biobert.parameters():
            param.requires_grad = False
            
        if vocab_size is None:
            vocab_size = self.biobert.config.vocab_size
        self.cnn_channels = (cnn_channels // nhead) * nhead
        self.conv = nn.Conv1d(self.biobert.config.hidden_size, self.cnn_channels, kernel_size=num_filters, padding=(num_filters - 1) // 2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.cnn_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(self.cnn_channels, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(self.cnn_channels, dropout)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.biobert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden_states = hidden_states.permute(0, 2, 1)
        cnn_output = self.conv(hidden_states).permute(0, 2, 1)
        cnn_output = self.pos_encoder(cnn_output)
        src_key_padding_mask = (attention_mask == 0).bool().to(device)
        transformer_output = self.transformer_encoder(cnn_output, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(transformer_output)

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered. No improvement for %d epochs.", self.patience)
                self.early_stop = True

def train_and_evaluate(trial, device, train_loader, test_loader):
    logging.info(f"Starting trial {trial.number} with parameters: {trial.params}")
    cnn_channels = trial.suggest_int("cnn_channels", 192, 768, step=96)
    nhead = trial.suggest_categorical("nhead", [6, 8, 12])
    cnn_channels = (cnn_channels // nhead) * nhead

    model = BioBERT_CNN_Transformer(
        cnn_channels=cnn_channels,
        num_filters=trial.suggest_int("num_filters", 3, 7, step=2),
        num_layers=trial.suggest_int("num_layers", 4, 8, step=2),
        dropout=trial.suggest_float("dropout", 0.1, 0.3),
        nhead=nhead,
        dim_feedforward=trial.suggest_int("dim_feedforward", 1024, 2048, step=512),
        vocab_size=tokenizer.vocab_size
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=trial.suggest_float('lr', 5e-5, 1e-4, log=True))
    early_stopping = EarlyStopping(patience=3, delta=0.01)
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    num_epochs = 2
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        logging.info(f"Epoch {epoch + 1} started.")

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Mixed precision training
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            
            # Scale gradients and backpropagate
            scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights with scaled gradients
            scaler.step(optimizer)
            scaler.update()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                
                # Use mixed precision for evaluation as well
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        epoch_duration = time.time() - start_time
        logging.info(f"Epoch {epoch + 1} ended. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.2f}s")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            break

    writer.add_scalar("Trial/FinalValLoss", avg_val_loss, trial.number)
    writer.flush()
    return avg_val_loss

def objective(trial):
    try:
        return train_and_evaluate(trial, device, train_loader, test_loader)
    except Exception as e:
        logging.error(f"Trial {trial.number} failed due to error: {str(e)}")
        logging.error(traceback.format_exc())
        return float('inf')

if __name__ == "__main__":
    # Logging Setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('output_file.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Starting training script")
    logger.info("GPU Available: %s", torch.cuda.is_available())
    logger.info("Using device: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    logger.info("Using mixed precision training with AMP")

    # Set Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction="minimize",
        study_name="medical_chatbot_tuning",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0)
    )
    study.optimize(objective, n_trials=150, show_progress_bar=True, gc_after_trial=True)

    logger.info(f"Best Hyperparameters: {study.best_params}")

    best_cnn_channels = study.best_params["cnn_channels"]
    best_nhead = study.best_params["nhead"]
    best_cnn_channels = (best_cnn_channels // best_nhead) * best_nhead

    final_model = BioBERT_CNN_Transformer(
        cnn_channels=best_cnn_channels,
        num_filters=study.best_params["num_filters"],
        num_layers=study.best_params["num_layers"],
        dropout=study.best_params["dropout"],
        nhead=best_nhead,
        dim_feedforward=study.best_params["dim_feedforward"],
        vocab_size=tokenizer.vocab_size
    ).to(device)

    torch.save(final_model.state_dict(), 'best_biobert_cnn_transformer.pth')
    tokenizer.save_pretrained("biobert_tokenizer")
    logger.info("Final model and tokenizer saved. TensorBoard logs are in runs/medical_chatbot")
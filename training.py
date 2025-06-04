import os
import sys
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import BertTokenizer, BertModel
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random
import numpy as np
import signal
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# ─── Setup & Logging ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

log_file = "logs/train_log.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler - supports UTF-8
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Console handler - fallback to stdout, avoid Unicode errors
console_handler = logging.StreamHandler(stream=sys.__stdout__)
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.handlers = []  # Clear existing handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Optional: redirect stdout to logger
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        if buf.strip():
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass
    
sys.stdout = StreamToLogger(logger, logging.INFO)

# ─── Signal Handling for Graceful Shutdown ──────────────────────────────────────
class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logger.info(f"\nReceived signal {signum}. Gracefully shutting down...")
        self.kill_now = True

def save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_val_loss, is_interrupt=False):
    """Save checkpoint with proper naming for different scenarios"""
    prefix = "interrupted" if is_interrupt else "epoch"
    checkpoint_path = f"checkpoints/{prefix}_{epoch}.pth"
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "interrupted": is_interrupt
    }, checkpoint_path)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

killer = GracefulKiller()


# ─── Optimized Dataset & Sampler ─────────────────────────────────────────────────
class MedicalChatbotDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, cache_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading tokenized dataset from cache: {cache_path}")
            cached = torch.load(cache_path)
            self.query_input_ids = cached["query_input_ids"]
            self.query_attention_mask = cached["query_attention_mask"]
            self.response_input_ids = cached["response_input_ids"]
            self.response_attention_mask = cached["response_attention_mask"]
            self.total_lengths = cached["total_lengths"]
            self.df = cached["df"]
            logger.info(f"Loaded cached dataset of size {len(self.df)}")
            return

        self.df = df.copy()
        logger.info("Precomputing tokenizations...")
        query_ids, query_masks = [], []
        response_ids, response_masks = [], []
        self.total_lengths = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
            q_enc = tokenizer(
                row["query"], truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt"
            )
            r_enc = tokenizer(
                row["response"], truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt"
            )

            query_ids.append(q_enc.input_ids.squeeze(0))
            query_masks.append(q_enc.attention_mask.squeeze(0))
            response_ids.append(r_enc.input_ids.squeeze(0))
            response_masks.append(r_enc.attention_mask.squeeze(0))

            q_len = (q_enc.input_ids != tokenizer.pad_token_id).sum().item()
            r_len = (r_enc.input_ids != tokenizer.pad_token_id).sum().item()
            self.total_lengths.append(q_len + r_len)

        self.query_input_ids = torch.stack(query_ids)
        self.query_attention_mask = torch.stack(query_masks)
        self.response_input_ids = torch.stack(response_ids)
        self.response_attention_mask = torch.stack(response_masks)
        self.total_lengths = torch.tensor(self.total_lengths)

        logger.info(f"Tokenization complete. Dataset size: {len(self.df)}")

        if cache_path:
            torch.save({
                "query_input_ids": self.query_input_ids,
                "query_attention_mask": self.query_attention_mask,
                "response_input_ids": self.response_input_ids,
                "response_attention_mask": self.response_attention_mask,
                "total_lengths": self.total_lengths,
                "df": self.df
            }, cache_path)
            logger.info(f"Saved tokenized dataset to cache: {cache_path}")

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "input_ids": self.query_input_ids[idx],
            "attention_mask": self.query_attention_mask[idx],
            "labels": self.response_input_ids[idx],
            "label_attention_mask": self.response_attention_mask[idx],
            "response_text": self.df.iloc[idx]["response"]
        }

class BioBERT_EncoderDecoder(nn.Module):
    def __init__(self, num_layers, dropout, nhead, dim_feedforward,
                 vocab_size, pad_token_id, max_len=256):
        super().__init__()
        self.encoder = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        d_model = self.encoder.config.hidden_size

        # Generate causal mask once and reuse - move to correct device in forward
        self.max_len = max_len
        self.register_buffer("causal_mask", 
                           torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # Add layer norm before output
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Improved initialization
        self.pad_token_id = pad_token_id
        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for embeddings and linear layers
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[self.pad_token_id].fill_(0)  # Zero pad token
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        B, T = decoder_input_ids.size()
        
        # Encoder
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = enc.last_hidden_state

        # Decoder embeddings with position encoding
        dec_emb = self.embedding(decoder_input_ids) + self.pos_embedding[:, :T, :]
        dec_emb = self.dropout(dec_emb)

        # Masks
        tgt_mask = self.causal_mask[:T, :T]
        tgt_key_padding_mask = (decoder_input_ids == self.pad_token_id)
        memory_key_padding_mask = (attention_mask == 0)

        # Decoder
        out = self.decoder(
            tgt=dec_emb, memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Layer norm + output projection
        out = self.layer_norm(out)
        return self.fc_out(out)

def compute_loss(logits, labels, pad_token_id, label_smoothing=0.1):
    """Optimized loss computation with label smoothing"""
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        reduction="mean",
        label_smoothing=label_smoothing
    )
    
    B, T, V = logits.size()
    return loss_fn(logits.reshape(-1, V), labels.reshape(-1))

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):  # Increased patience and delta
        self.patience, self.delta = patience, delta
        self.best_loss, self.counter = None, 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
# ─── Optimized Sample Prediction Logger ─────────────────────────────────────────
def log_sample_predictions(model, loader, tokenizer, device, epoch, logger, num_samples=3, max_gen_len=100):
    model.eval()
    logged_samples = 0

    start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    end_token_ids = [tokenizer.sep_token_id, tokenizer.pad_token_id]

    with torch.no_grad():
        for batch in loader:
            if logged_samples >= num_samples:
                break

            in_ids = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            batch_size = in_ids.size(0)

            for i in range(min(num_samples - logged_samples, batch_size)):
                input_ids = in_ids[i].unsqueeze(0)
                attention_mask = amask[i].unsqueeze(0)
                
                generated_ids = [start_token_id]
                for _ in range(max_gen_len):
                    dec_in = torch.tensor([generated_ids], device=device)
                    logits = model(input_ids, attention_mask, dec_in)
                    next_token = logits[:, -1, :].argmax(-1).item()
                    
                    if next_token in end_token_ids:
                        break
                    generated_ids.append(next_token)

                inp = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                prd = tokenizer.decode(generated_ids, skip_special_tokens=True)
                tgt = tokenizer.decode(batch["labels"][i], skip_special_tokens=True)

                logger.info(f"\n{'='*50}\nEPOCH {epoch} - SAMPLE {logged_samples+1}:\n{'='*50}")
                logger.info(f"INPUT: {inp[:150]}...")
                logger.info(f"PRED : {prd[:150]}...")
                logger.info(f"TRUE : {tgt[:150]}...")

                logged_samples += 1
            break

# ─── Main Training Script ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hyperparameters
    epochs = 100
    batch_size = 4
    max_length = 256
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 1) Tokenizer & Data
    logger.info("Loading tokenizer and data...")
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '[CLS]'})
    
    train_df = pd.read_csv("processed_data/train_data.csv")#.sample(frac=0.001, random_state=42).reset_index(drop=True)
    test_df = pd.read_csv("processed_data/test_data.csv")#.sample(frac=0.001, random_state=42).reset_index(drop=True)

    train_ds = MedicalChatbotDataset(train_df, tokenizer, max_length, cache_path="tokenized_train.pt")
    test_ds = MedicalChatbotDataset(test_df, tokenizer, max_length, cache_path="tokenized_train.pt")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # 2) Model, Optimizer, Scheduler
    logger.info("Initializing model...")
    model = BioBERT_EncoderDecoder(
        num_layers=4, dropout=0.1, nhead=8,
        dim_feedforward=2048,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        max_len=max_length
    ).to(device)

    # Optimized optimizer settings
    optimizer = AdamW(
        model.parameters(), lr=5e-5, weight_decay=0.01,
        betas=(0.9, 0.999), eps=1e-8
    )
    
    scheduler = OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1, anneal_strategy='cos'
    )
    
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=7, delta=0.001)
    best_val_loss = float("inf")

    # 3) Resume from checkpoint if available
    ckpts = sorted(f for f in os.listdir("checkpoints") if f.endswith(".pth"))
    start_epoch = 1
    if ckpts:
        logger.info(f"Found checkpoint: {ckpts[-1]}")
        cp = torch.load(os.path.join("checkpoints", ckpts[-1]), map_location=device)
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        try:
            scaler.load_state_dict(cp["scaler_state_dict"])
        except:
            pass
        scheduler.load_state_dict(cp["scheduler_state_dict"])
        best_val_loss = cp["best_val_loss"]
        start_epoch = cp["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch-1}")

    # 4) Training Loop
    logger.info("Starting training...")
    try:
        accum_steps = 4
        optimizer.zero_grad()

        for epoch in range(start_epoch, epochs + 1):
            if killer.kill_now:
                logger.info("Interrupt detected at start of epoch. Saving checkpoint...")
                save_checkpoint(model, optimizer, scaler, scheduler, epoch-1, best_val_loss, is_interrupt=True)
                break

            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                if killer.kill_now:
                    logger.info(f"Interrupt detected during epoch {epoch}, batch {batch_idx}. Saving checkpoint...")
                    save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_val_loss, is_interrupt=True)
                    break

                in_ids = batch["input_ids"].to(device, non_blocking=True)
                amask = batch["attention_mask"].to(device, non_blocking=True)
                labs = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                    dec_in = labs[:, :-1]
                    dec_lab = labs[:, 1:]
                    logits = model(in_ids, amask, dec_in)
                    loss = compute_loss(logits, dec_lab, tokenizer.pad_token_id, 0.1)

                # Scale loss for accumulation
                loss = loss / accum_steps
                scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                train_loss += loss.item() * accum_steps  # undo division for avg tracking

                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item() * accum_steps:.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                    })

            avg_train = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch} completed. | Avg Train Loss: {avg_train:.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    if killer.kill_now:
                        logger.info(f"Interrupt detected during validation of epoch {epoch}. Saving checkpoint...")
                        save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_val_loss, is_interrupt=True)
                        break

                    in_ids = batch["input_ids"].to(device, non_blocking=True)
                    amask = batch["attention_mask"].to(device, non_blocking=True)
                    labs = batch["labels"].to(device, non_blocking=True)

                    with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        dec_in = labs[:, :-1]
                        dec_lab = labs[:, 1:]
                        logits = model(in_ids, amask, dec_in)
                        loss = compute_loss(logits, dec_lab, tokenizer.pad_token_id, 0.1)

                    val_loss += loss.item()

            if killer.kill_now:
                break

            avg_val = val_loss / len(test_loader)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss
                }, f"checkpoints/best_epoch_{epoch}.pth")
                logger.info(f"New best model saved (val_loss: {avg_val:.4f})")

            if epoch % 5 == 0:
                log_sample_predictions(model, test_loader, tokenizer, device, epoch, logger, 2)

            if early_stopping(avg_val):
                logger.info("Early stopping triggered.")
                save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_val_loss, is_interrupt=False)
                break
                
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught! Saving checkpoint...")
        save_checkpoint(model, optimizer, scaler, scheduler, 
                       epoch if 'epoch' in locals() else start_epoch, 
                       best_val_loss, is_interrupt=True)
        logger.info("Checkpoint saved. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        logger.info("Saving emergency checkpoint...")
        save_checkpoint(model, optimizer, scaler, scheduler, 
                       epoch if 'epoch' in locals() else start_epoch, 
                       best_val_loss, is_interrupt=True)
        raise

    # 5) Final model save
    os.makedirs("final_model", exist_ok=True)
    torch.save(model.state_dict(), "final_model/biobert_transformer_final.pth")
    tokenizer.save_pretrained("final_model/")
    logger.info("Training complete - model and tokenizer saved.")
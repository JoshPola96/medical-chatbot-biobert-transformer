import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os

# Streamlit config
st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer - Fixed path to match training script
@st.cache_resource
def load_tokenizer():
    tokenizer_path = "final_model/"
    if os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        st.warning("Custom tokenizer not found. Using original BioBERT tokenizer.")
        tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        # Add BOS token if not present (matching training script)
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({'bos_token': '[CLS]'})
    return tokenizer

tokenizer = load_tokenizer()

# Updated Model definition to match training script exactly
class BioBERT_EncoderDecoder(nn.Module):
    def __init__(self, num_layers, dropout, nhead, dim_feedforward,
                 vocab_size, pad_token_id, max_len=256):
        super().__init__()
        self.encoder = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        d_model = self.encoder.config.hidden_size

        # Generate causal mask once and reuse - matching training script
        self.max_len = max_len
        self.register_buffer("causal_mask", 
                           torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

        # Updated decoder layer with norm_first=True for better stability
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Updated embedding and layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # Add layer norm before output
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Store pad_token_id
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

# Load model with updated hyperparameters from training script
@st.cache_resource
def load_model():
    # Updated hyperparameters to match training script
    model = BioBERT_EncoderDecoder(
        num_layers=4,  # Updated from 6
        dropout=0.1,   # Updated from 0.16504607952750663
        nhead=8,       # Updated from 6
        dim_feedforward=2048,  # Updated from 1024
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,  # Added pad_token_id parameter
        max_len=256    # Updated from default
    ).to(device)
    
    # Load the saved state dict - FIXED TO HANDLE CHECKPOINT FORMAT
    model_path = "final_model/biobert_transformer_final.pth"
    checkpoint_paths = []
    
    # Look for checkpoint files if final model doesn't exist
    if not os.path.exists(model_path):
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoint_files:
                # Sort by modification time and get the latest
                checkpoint_paths = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
                checkpoint_paths.sort(key=os.path.getmtime, reverse=True)
                model_path = checkpoint_paths[0]
                st.info(f"Using checkpoint: {os.path.basename(model_path)}")
    
    if os.path.exists(model_path):
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if it's a checkpoint file (contains nested structure) or direct state dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # It's a checkpoint file - extract the model state dict
                model.load_state_dict(checkpoint["model_state_dict"])
                epoch = checkpoint.get("epoch", "unknown")
                val_loss = checkpoint.get("best_val_loss", "unknown")
                st.success(f"Model loaded successfully from checkpoint! (Epoch: {epoch}, Val Loss: {val_loss:.4f})")
            else:
                # It's a direct state dict
                model.load_state_dict(checkpoint)
                st.success("Model loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("Please check if your model file is compatible with the current model architecture.")
            st.stop()
    else:
        st.error(f"No model file found. Looked for:")
        st.error(f"1. {model_path}")
        if checkpoint_paths:
            st.error("2. Checkpoint files in 'checkpoints/' directory")
        st.error("Please ensure you have trained and saved a model first.")
        st.stop()
    
    model.eval()
    return model

model = load_model()

# Updated beam search to match the new architecture
def beam_search(model, input_ids, attention_mask, beam_size=3, max_length=1024):
    # Initial encoder pass
    encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    memory = encoder_outputs.last_hidden_state
    
    # Start with BOS token - updated to use proper token
    start_token = tokenizer.cls_token_id or tokenizer.bos_token_id
    beams = [(torch.tensor([[start_token]], device=device), 0)]
    
    for _ in range(max_length):
        new_beams = []
        
        for beam_tokens, beam_score in beams:
            # If the last token is EOS, keep this beam as is
            if beam_tokens[0, -1].item() == tokenizer.sep_token_id:
                new_beams.append((beam_tokens, beam_score))
                continue
                
            # Forward pass for this beam
            beam_len = beam_tokens.shape[1]
            
            # Updated to match new architecture
            dec_emb = model.embedding(beam_tokens) + model.pos_embedding[:, :beam_len, :]
            dec_emb = model.dropout(dec_emb)
            
            # Use the registered buffer causal mask
            tgt_mask = model.causal_mask[:beam_len, :beam_len]
            tgt_key_padding_mask = (beam_tokens == model.pad_token_id)
            memory_key_padding_mask = (attention_mask == 0)
            
            out = model.decoder(
                tgt=dec_emb, memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Apply layer norm and get logits
            out = model.layer_norm(out)
            logits = model.fc_out(out[:, -1:, :])
            probs = torch.nn.functional.log_softmax(logits.squeeze(1), dim=-1)
            
            # Get top k probabilities and corresponding token IDs
            top_probs, top_indices = probs.topk(beam_size)
            
            for i in range(beam_size):
                token_id = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_tokens = torch.cat([beam_tokens, token_id], dim=1)
                new_score = beam_score + top_probs[0, i].item()
                new_beams.append((new_tokens, new_score))
        
        # Select top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Check if all beams end with EOS
        if all(beam[0][0, -1].item() == tokenizer.sep_token_id for beam in beams):
            break
    
    # Return the highest scoring beam
    return beams[0][0]

# Generate response function remains mostly the same
def generate_response(input_text, max_length=256):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        # Use beam search for better generation
        output_ids = beam_search(model, input_ids, attention_mask, beam_size=3, max_length=max_length)
        
        # Convert IDs to tokens
        response_tokens = tokenizer.convert_ids_to_tokens(output_ids[0].tolist())
        
    # Clean the output
    response_text = tokenizer.convert_tokens_to_string(response_tokens).strip()
    
    # Remove special tokens and any text after [SEP]
    if tokenizer.sep_token in response_text:
        response_text = response_text.split(tokenizer.sep_token)[0].strip()
    
    return response_text

# UI
st.title("🏥 Medical Chatbot with BioBERT + Transformer")
st.markdown("Ask a medical-related question and receive an AI-generated response.")

# Display model info
with st.expander("Model Information"):
    st.write(f"**Device:** {device}")
    st.write(f"**Vocabulary Size:** {tokenizer.vocab_size}")
    st.write(f"**Model Architecture:** BioBERT Encoder + Transformer Decoder")
    st.write(f"**Layers:** 4 decoder layers, 8 attention heads")

user_input = st.text_area("Your question:", height=100, placeholder="Enter your medical question here...")

col1, col2 = st.columns([1, 1])
with col1:
    generate_btn = st.button("Generate Response", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.rerun()

if generate_btn and user_input.strip():
    with st.spinner("Generating response..."):
        try:
            response = generate_response(user_input)
            st.markdown("**Response:**")
            if response and response.strip():
                st.success(response)
            else:
                st.warning("Sorry, I could not generate a meaningful response. Please try rephrasing your question.")
        except Exception as e:
            st.error(f"An error occurred during generation: {str(e)}")
            st.info("Please check if your model files are properly saved and compatible.")

elif generate_btn:
    st.warning("Please enter a question before generating a response.")

# Footer
st.markdown("---")
st.markdown("*This is an AI-generated response for educational purposes only. Always consult with healthcare professionals for medical advice.*")
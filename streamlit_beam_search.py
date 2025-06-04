import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Streamlit config
st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("final_model - Copy/biobert_tokenizer_final")

tokenizer = load_tokenizer()

# Model definition - using the BioBERT_EncoderDecoder from your training code
class BioBERT_EncoderDecoder(nn.Module):
    def __init__(self, num_layers, dropout, nhead, dim_feedforward, vocab_size, max_len=256):
        super().__init__()
        self.encoder = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        
        d_model = self.encoder.config.hidden_size
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
    
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # Encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = encoder_outputs.last_hidden_state  # shape: [B, T, H]
        
        # Decoder
        B, T = decoder_input_ids.shape
        positions = self.pos_embedding[:, :T, :]  # shape: [1, T, H]
        dec_emb = self.embedding(decoder_input_ids) + positions  # shape: [B, T, H]
        dec_emb = self.dropout(dec_emb)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(dec_emb.device).to(torch.bool)
        
        out = self.transformer_decoder(
            tgt=dec_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(attention_mask == 0),
            tgt_key_padding_mask=(decoder_input_ids == tokenizer.pad_token_id),
        )
        
        logits = self.fc_out(out)  # shape: [B, T, V]
        return logits

# Load model
@st.cache_resource
def load_model():
    # Define model with the same hyperparameters from your training
    model = BioBERT_EncoderDecoder(
        num_layers=6,
        dropout=0.16504607952750663,
        nhead=6,
        dim_feedforward=1024,
        vocab_size=tokenizer.vocab_size
    ).to(device)
    
    # Load the saved state dict
    model.load_state_dict(torch.load("final_model - Copy/biobert_transformer_final.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Beam search for better generation
def beam_search(model, input_ids, attention_mask, beam_size=3, max_length=128):
    # Initial encoder pass
    encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    memory = encoder_outputs.last_hidden_state
    
    # Start with BOS token
    start_token = tokenizer.cls_token_id
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
            positions = model.pos_embedding[:, :beam_len, :]
            dec_emb = model.embedding(beam_tokens) + positions
            dec_emb = model.dropout(dec_emb)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(beam_len).to(dec_emb.device).to(torch.bool)
            
            out = model.transformer_decoder(
                tgt=dec_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=(attention_mask == 0),
                tgt_key_padding_mask=(beam_tokens == tokenizer.pad_token_id)
            )
            
            # Get logits for the next token
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

# Generate response
def generate_response(input_text, max_length=128):
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

user_input = st.text_area("Your question:", height=100)

if st.button("Generate Response") and user_input.strip():
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.markdown("**Response:**")
    st.success(response if response else "Sorry, I could not generate a response.")
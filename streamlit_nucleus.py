import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="wide")

import torch
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel

# Load the saved model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BioBERT tokenizer
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("final_model/biobert_tokenizer_final/")

tokenizer = load_tokenizer()

# Define your model architecture to match the one used during training
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
    
    def forward(self, input_ids, attention_mask, decoder_input_ids=None):
        # Encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = encoder_outputs.last_hidden_state  # shape: [B, T, H]
        
        if decoder_input_ids is None:
            # For inference when we don't have decoder inputs yet
            return memory
            
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

# Load the model weights from the saved file
@st.cache_resource
def load_model():
    model = BioBERT_EncoderDecoder(
        num_layers=6,
        dropout=0.16504607952750663,
        nhead=6,
        dim_feedforward=1024,
        vocab_size=tokenizer.vocab_size
    ).to(device)
    model.load_state_dict(torch.load('final_model/biobert_transformer_final.pth', map_location=device))
    model.eval()
    return model

# Load the model
model = load_model()

# Function to preprocess the input text
def preprocess_input(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    return inputs

# Clean response generation with only nucleus sampling
def generate_response(input_text, max_length=150, top_p=0.92, temperature=0.7):
    inputs = preprocess_input(input_text)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        # Encode the input sequence first
        encoder_output = model(input_ids, attention_mask)
        
        # Start with special token (CLS is often used as BOS in BERT)
        decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
        generated_tokens = []
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Get positions for current decoder sequence length
            seq_len = decoder_input_ids.size(1)
            positions = model.pos_embedding[:, :seq_len, :]
            
            # Embed decoder input
            dec_emb = model.embedding(decoder_input_ids) + positions
            dec_emb = model.dropout(dec_emb)
            
            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device).to(torch.bool)
            
            # Decoder forward pass
            # Ensure memory_key_padding_mask has proper dimensions
            memory_pad_mask = None
            if attention_mask is not None:
                memory_pad_mask = (attention_mask == 0)
                
            # Ensure tgt_key_padding_mask has proper dimensions
            tgt_pad_mask = None
            if decoder_input_ids is not None:
                tgt_pad_mask = (decoder_input_ids == tokenizer.pad_token_id)
                
            decoder_output = model.transformer_decoder(
                tgt=dec_emb,
                memory=encoder_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )
            
            # Get logits for the next token
            next_token_logits = model.fc_out(decoder_output[:, -1:])
            # Ensure consistent dimensions by explicitly reshaping
            next_token_logits = next_token_logits.squeeze(1)  # Remove sequence dimension to get [batch, vocab_size]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the chosen token to our output sequence
            generated_tokens.append(next_token.item())
            
            # Stop if we generate an end token
            if next_token.item() in [tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
                
            # Fix the dimension mismatch: ensure next_token is shaped correctly [batch_size, seq_len]
            # The issue is that next_token is [batch_size, 1] but we need it to be [batch_size, 1]
            # Make sure it's exactly shaped as [1, 1] for proper concatenation
            next_token = next_token.view(1, 1)  # Explicitly reshape to [1, 1]
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
    
    # Convert generated tokens to text
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response_text

# Streamlit UI

# Add title and description
st.title("🏥 Medical Chatbot")
st.markdown("""
This AI assistant can answer your medical questions. Please note that this is for 
informational purposes only and should not replace professional medical advice.
""")

# Create a container for the chat interface
chat_container = st.container()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display the chat history
with chat_container:
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"**You**: {content}")
        else:
            st.markdown(f"**Medical Assistant**: {content}")
    
    st.markdown("---")

# Create columns for advanced options
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Response Creativity (Temperature)", min_value=0.1, max_value=1.5, value=0.7, step=0.1, 
                         help="Higher values make responses more creative but potentially less accurate")

with col2:
    top_p = st.slider("Nucleus Sampling (Top-p)", min_value=0.5, max_value=1.0, value=0.92, step=0.01,
                    help="Controls diversity by limiting token selection to the most likely ones")

# Input area for user questions
user_input = st.text_area("Ask a medical question:", height=100)

# Submit button
if st.button("Get Response"):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show a spinner while generating the response
        with st.spinner("Generating response..."):
            try:
                response = generate_response(user_input, top_p=top_p, temperature=temperature)
                
                # Check if we got a valid response
                if not response or len(response.strip()) < 5:
                    response = "I couldn't generate a proper response. Please try rephrasing your question."
                    
            except Exception as e:
                # Return the error message directly
                response = f"Error during generation: {str(e)}"
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Refresh the page to show the updated chat
        st.rerun()
    else:
        st.warning("Please enter a question.")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*This is a research prototype. Always consult with healthcare professionals for medical advice.*")
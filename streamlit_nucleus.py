import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="wide")

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import glob

# Load the saved model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BioBERT tokenizer
@st.cache_resource
def load_tokenizer():
    # Try to load from final_model directory first, then fallback to pretrained
    tokenizer_path = "final_model/"
    if not os.path.exists(tokenizer_path):
        # Fallback to the original tokenizer used in training
        return BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure BOS token is available (matching training setup)
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '[CLS]'})
    
    return tokenizer

tokenizer = load_tokenizer()

# Define your model architecture to EXACTLY match the training one
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

# Load the model weights from the saved file
@st.cache_resource
def load_model():
    # Create model with exact same parameters as training
    model = BioBERT_EncoderDecoder(
        num_layers=4,  # Match training parameters
        dropout=0.1,
        nhead=8,
        dim_feedforward=2048,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        max_len=256
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

# Load the model
model = load_model()

# Function to preprocess the input text
def preprocess_input(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    return inputs

# Improved response generation matching training setup
def generate_response(input_text, max_length=1024, top_p=0.92, temperature=0.7):
    inputs = preprocess_input(input_text)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Start with CLS token (matching training setup)
    start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    end_token_ids = [tokenizer.sep_token_id, tokenizer.pad_token_id]
    if tokenizer.eos_token_id is not None:
        end_token_ids.append(tokenizer.eos_token_id)
    
    with torch.no_grad():
        generated_ids = [start_token_id]
        
        for _ in range(max_length):
            # Prepare decoder input
            decoder_input_ids = torch.tensor([generated_ids], device=device)
            
            try:
                # Forward pass through the model
                logits = model(input_ids, attention_mask, decoder_input_ids)
                
                # Get logits for the next token (last position)
                next_token_logits = logits[:, -1, :].squeeze(0)  # [vocab_size]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if we generate an end token
                if next_token in end_token_ids:
                    break
                    
                generated_ids.append(next_token)
                
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
                break
    
    # Decode the generated tokens (skip the start token)
    try:
        response_text = tokenizer.decode(generated_ids[1:], skip_special_tokens=True).strip()
        
        # Clean up the response
        if not response_text:
            response_text = "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
        
        return response_text
    except Exception as e:
        return f"Error decoding response: {str(e)}"

# Streamlit UI

# Add title and description
st.title("🏥 Medical Chatbot")
st.markdown("""
This AI assistant can answer your medical questions using a BioBERT-based model. 
Please note that this is for informational purposes only and should not replace professional medical advice.
""")

# Display model info
with st.expander("Model Information"):
    st.write(f"**Device**: {device}")
    st.write(f"**Tokenizer Vocabulary Size**: {tokenizer.vocab_size}")
    st.write(f"**Model Parameters**: Encoder-Decoder with BioBERT encoder")

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

max_length = st.slider("Maximum Response Length", min_value=1, max_value=1024, value=100, step=10,
                      help="Maximum number of tokens to generate")

# Input area for user questions
user_input = st.text_area("Ask a medical question:", height=100, 
                         placeholder="e.g., What are the symptoms of diabetes?")

# Submit button
if st.button("Get Response", type="primary"):
    if user_input.strip():
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show a spinner while generating the response
        with st.spinner("Generating response..."):
            try:
                response = generate_response(
                    user_input, 
                    max_length=max_length,
                    top_p=top_p, 
                    temperature=temperature
                )
                
                # Check if we got a valid response
                if not response or len(response.strip()) < 3:
                    response = "I couldn't generate a proper response. Please try rephrasing your question or adjusting the generation parameters."
                    
            except Exception as e:
                response = f"Error during generation: {str(e)}"
                st.error(response)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Refresh the page to show the updated chat
        st.rerun()
    else:
        st.warning("Please enter a question.")

# Add control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if st.button("Example Questions"):
        examples = [
            "What are the symptoms of hypertension?",
            "How is diabetes diagnosed?",
            "What causes migraine headaches?",
            "What are the risk factors for heart disease?"
        ]
        st.write("**Try these example questions:**")
        for example in examples:
            st.write(f"• {example}")

# Footer
st.markdown("---")
st.markdown("*This is a research prototype powered by BioBERT. Always consult with healthcare professionals for medical advice.*")
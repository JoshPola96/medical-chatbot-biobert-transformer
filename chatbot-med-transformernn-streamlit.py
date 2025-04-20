import streamlit as st
import tensorflow as tf
import keras
from transformers import BertTokenizer, AutoConfig, TFBertModel
import re
import os


# Load pre-trained BioBERT tokenizer
TOKENIZER_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

# BioBertEncoder class definition (unchanged)
@keras.utils.register_keras_serializable(package="Custom")
class BioBertEncoder(keras.layers.Layer):
    def __init__(self, bert_model_name, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.trainable = trainable
        self.bert_model = None
        
    def build(self, input_shape):
        self.bert_config = AutoConfig.from_pretrained(self.bert_model_name)
        self.bert_model = TFBertModel.from_pretrained(
            self.bert_model_name,
            config=self.bert_config,
            from_pt=True
        )
        self.bert_model.trainable = self.trainable
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )
        return outputs.last_hidden_state
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "bert_model_name": self.bert_model_name,
            "trainable": self.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        bert_model_name = config["bert_model_name"]
        trainable = config["trainable"]
        return cls(bert_model_name=bert_model_name, trainable=trainable)    

# BioBERT Preprocessor Class (unchanged)
class BioBERTPreprocessor:
    def __init__(self, model_name=TOKENIZER_MODEL_NAME):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def _clean_medical_text(self, text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', ' - ', text)
        text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])\.([A-Za-z])\.', r'\1\2', text)
        text = re.sub(r'[^A-Za-z0-9\s\-\+\u00B0%/\(\)\[\]\{\}\u00B1\u2192\u2190\u2194\u2265\u2264=\u2260\u03BC\u03B1\u03B2\u03B3\.]', ' ', text)
        return text.strip()

    def prepare_input(self, query, max_length):
        clean_query = self._clean_medical_text(query)
        query_encodings = self.tokenizer(
            clean_query,
            padding='max_length',
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        return query_encodings

# Initialize Preprocessor
MAX_LENGTH = 256
preprocessor = BioBERTPreprocessor()

# Load Fine-Tuned Model   
def load_fine_tuned_model():
    model_path = 'E:/personal/Code/Irohub_DS/10 - Projects/4 - Chatbot_Medical_Advice/model/transformernn_model.keras'
    if os.path.exists(model_path):
        custom_objects = {
            'BioBertEncoder': BioBertEncoder
        }
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            st.success("Fine-tuned model loaded successfully.")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.error(f"Fine-tuned model not found at '{model_path}'. Please check the path.")
        return None

# Initialize Model (load once)
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = load_fine_tuned_model()

# Generate BioBERT Embeddings (Preprocessed Input)
def get_biobert_embeddings(query):
    query_encodings = preprocessor.prepare_input(query, MAX_LENGTH)
    model_input = {
        'query_input_ids': query_encodings['input_ids'],
        'query_attention_mask': query_encodings['attention_mask']
    }
    return model_input

def sample_from_probabilities(probs, temperature=1.0, top_k=None):
    # Apply temperature scaling
    probs = probs / temperature

    # Apply top-k sampling (if specified)
    if top_k is not None:
        top_k_values, top_k_indices = tf.math.top_k(probs, k=top_k)
        probs = -tf.math.inf * tf.ones_like(probs)  # Set all probabilities to -inf
        probs = tf.tensor_scatter_nd_update(probs, tf.expand_dims(top_k_indices, axis=-1), top_k_values)

    # Sample from the probability distribution
    sampled_token_id = tf.random.categorical(probs, num_samples=1)
    return sampled_token_id.numpy().flatten()


# Generate Response from Fine-Tuned Model using Sampling
def get_trained_model_response(query):
    trained_model = st.session_state.trained_model
    if trained_model is None:
        return "Error: Fine-tuned model not loaded."

    try:
        # Step 1: Get BioBERT embeddings for the query
        model_input = get_biobert_embeddings(query)
        if model_input is None or 'query_input_ids' not in model_input:
            return "Error: Model input is missing required keys."

        # Extract the input IDs and generate attention mask dynamically
        query_input_ids = tf.convert_to_tensor(model_input['query_input_ids'])

        # Generate attention mask based on input_ids (0 for padding, 1 for non-padding)
        query_attention_mask = tf.cast(query_input_ids != tokenizer.pad_token_id, tf.int32)

        # Step 2: Getting model response
        model_response = trained_model([query_input_ids, query_attention_mask])  # Passing inputs as tensors

        # Now, model_response contains probabilities, not logits
        probs = model_response[0]  # Assuming the response is [batch_size, seq_len, vocab_size]

        # Sample from the probabilities using top-k sampling or other techniques
        sampled_token_ids = sample_from_probabilities(probs[:, -1, :], temperature=1.0, top_k=5)  # Sampling the last token

        # Decode the predicted token IDs to get the response text
        response_text = tokenizer.decode(sampled_token_ids, skip_special_tokens=True)

        return response_text

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.title("Medical Chatbot with Fine-Tuned Model")
query = st.text_area("Enter the medical query:")

if st.button('Generate Response'):
    if query:
        response = get_trained_model_response(query)
        st.write(f"Model Response: {response}")
    else:
        st.warning("Please enter a query.")

import streamlit as st
import tensorflow as tf
import keras
from transformers import BertTokenizer, AutoConfig, TFBertModel
import re
import os

# Load pre-trained BioBERT tokenizer
TOKENIZER_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

# BioBertEncoder class definition
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
        # This method is added for deserialization
        bert_model_name = config["bert_model_name"]
        trainable = config["trainable"]
        return cls(bert_model_name=bert_model_name, trainable=trainable)    

# BioBERT Preprocessor Class
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
            print("Model loaded:", model.summary())  # Debug: print model summary
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")  # Debug: print the error message
            return None
    else:
        st.error(f"Fine-tuned model not found at '{model_path}'. Please check the path.")
        print(f"Fine-tuned model not found at '{model_path}'. Please check the path.")  # Debug
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

# Generate Response from Fine-Tuned Model
def get_trained_model_response(query):
    trained_model = st.session_state.trained_model
    if trained_model is None:
        return "Error: Fine-tuned model not loaded."
    
    try:
        # Get preprocessed input embeddings for the query
        model_input = get_biobert_embeddings(query)

        # Ensure that the model_input is not None and has all expected keys
        if model_input is None or 'query_input_ids' not in model_input or 'query_attention_mask' not in model_input:
            return "Error: Model input is missing required keys."

        # Extract the inputs from the dictionary
        query_input_ids = model_input['query_input_ids']
        query_attention_mask = model_input['query_attention_mask']

        # Call model to get response (predict)
        model_response = trained_model({
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask
        })

        # Extract logits (output of the model)
        logits = model_response  # Model output is already in logits
        if logits is None:
            return "Error: Logits are None."
        
        # Apply argmax to get the predicted token IDs (assuming batch_size=1)
        predicted_token_ids = tf.argmax(logits, axis=-1)    
        
        predicted_token_ids = predicted_token_ids.numpy()[0]  # Get the token IDs for the first sample        

        # Decode the predicted token IDs to get the response text
        response_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)

        return response_text

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"  # Return the error message for debugging

# Streamlit UI
st.title("Medical Chatbot with Fine-Tuned Model")
query = st.text_area("Enter the medical query:")

if st.button('Generate Response'):
    if query:
        response = get_trained_model_response(query)
        st.write(f"Model Response: {response}")
    else:
        st.warning("Please enter a query.")

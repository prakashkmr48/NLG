import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Function to load GPT-2 model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

# Function for text generation using GPT-2
@st.cache(allow_output_mutation=True)
def generate_text(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Streamlit app
st.title("GPT-2 Text Generation App")

# Load GPT-2 model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Get user input prompt
user_prompt = st.text_input("Enter your prompt:")

if st.button("Generate Text"):
    # Generate response
    generated_response = generate_text(user_prompt, model, tokenizer)

    # Display generated response
    st.subheader("Generated Response:")
    st.write(generated_response)

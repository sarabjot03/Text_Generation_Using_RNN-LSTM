import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('RNN_LSTM_Model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Function to generate text
def generate_text(seed_text, model, tokenizer, max_sequence_len, num_words_to_generate, temperature=1.0):
    for _ in range(num_words_to_generate):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        
        # Apply temperature scaling
        predicted_probs = np.asarray(predicted_probs).astype('float64')
        predicted_probs = np.log(predicted_probs + 1e-8) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)
        
        # Randomly select the next word based on the probabilities
        predicted = np.random.choice(range(len(predicted_probs[0])), p=predicted_probs[0])
        
        # Convert the predicted index back to the word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    
    return seed_text

# Streamlit App
st.title("Shakespeare Text Generator")

# Input for the seed text
seed_text = st.text_input("Enter a seed text:", value="Shall I compare thee to a summer's day")

# Slider to choose the number of words to generate
num_words = st.slider("Number of words to generate:", min_value=10, max_value=100, value=50)

# Slider for temperature control
temperature = st.slider("Temperature (controls randomness):", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Button to generate text
if st.button("Generate Text"):
    generated_text = generate_text(seed_text, model, tokenizer, max_sequence_len=50, num_words_to_generate=num_words, temperature=temperature)
    st.subheader("Generated Text:")
    st.write(generated_text)
import streamlit as st

# Import your prediction function from the appropriate module
# Make sure `predict` and any necessary utilities are imported from your model handling script.
from main import predict  # Adjust the import based on where the prediction function is defined
#dont
# Define the list of models and their corresponding names in the backend
model_options = {
    "BERT": "BERTClass",
    "BERT with Attention": "BERTWithAttention",
    "GPT-2": "GPT2class",
    "RoBERTa": "RobertaClass",
    "BERT with Bidirectional LSTM": "BERTLSTM",
    "BERT with CNN": "BERTCNN",
    "BERT with Average": "BERTaverage"
}

# Streamlit app title
st.title("Toxicity Classification")

# Dropdown for model selection
model_choice = st.selectbox("Select a model for classification:", list(model_options.keys()))

# User input
user_input = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if user_input:
        with st.spinner("Classifying..."):
            try:
                # Call the prediction function directly
                result = predict(text=user_input, model_name=model_options[model_choice])

                # Handle the result
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"Prediction: {result['prediction']}")
                    st.info(f"Confidence: {result['confidence']:.4f}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter some text to classify.")

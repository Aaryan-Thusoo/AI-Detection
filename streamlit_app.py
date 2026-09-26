import streamlit as st
import pickle
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import sys

sys.path.append('pipeline/helpers')
from pipeline.helpers.searching import (
    word_count, sentence_count, punctuation_count, char_length,
    avg_word_length, uppercase_ratio, digit_ratio, url_count,
    email_count, vocabulary_diversity, check_scam_words
)

st.set_page_config(page_title="Fraud Email Detector", layout="wide")

# Title
st.title("🚨 Fraud Email Detection & Explanation")
st.markdown("Powered by XGBoost + Fine-tuned LLM")

# Sidebar for model info
with st.sidebar:
    st.header("Model Info")
    st.write("**Fraud Detector:** XGBoost (82.6% accuracy)")
    st.write("**Explainer:** TinyLlama 1.1B (fine-tuned)")


# Load models (cache for performance)
@st.cache_resource
def load_models():
    # Load XGBoost
    with open('pipeline/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # Load fine-tuned LLM
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
        "./finetune/mistral-fraud-adapter",
        device_map="auto"
    )

    return xgb_model, tokenizer, finetuned_model


try:
    xgb_model, tokenizer, llm_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# Email input
email_text = st.text_area(
    "📧 Enter an email to analyze:",
    height=200,
    placeholder="Paste the email text here..."
)

if st.button("🔍 Analyze Email", use_container_width=True):
    if not email_text:
        st.warning("Please enter an email to analyze")
    elif not models_loaded:
        st.error("Models not loaded. Check the sidebar error.")
    else:
        # Extract features
        features = {
            'word_count': word_count(email_text),
            'sentence_count': sentence_count(email_text),
            'char_length': char_length(email_text),
            'avg_word_length': avg_word_length(email_text),
            'uppercase_ratio': uppercase_ratio(email_text),
            'digit_ratio': digit_ratio(email_text),
            'url_count': url_count(email_text),
            'email_count': email_count(email_text),
            'vocabulary_diversity': vocabulary_diversity(email_text),
            'scam_words_count': check_scam_words(email_text),
        }

        # Add punctuation features
        punct = punctuation_count(email_text)
        features.update({
            f'punctuation_{k}': v for k, v in punct.items()
        })

        # Convert to dataframe for XGBoost
        X_input = pd.DataFrame([features])

        # Fraud prediction
        fraud_pred = xgb_model.predict(X_input)[0]
        fraud_prob = xgb_model.predict_proba(X_input)[0]

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Fraud Classification")
            if fraud_pred == 1:
                st.error("⚠️ **FRAUDULENT EMAIL**")
                confidence = fraud_prob[1] * 100
            else:
                st.success("✓ **LEGITIMATE EMAIL**")
                confidence = fraud_prob[0] * 100

            st.metric("Confidence", f"{confidence:.1f}%")

        with col2:
            st.subheader("📈 Feature Analysis")
            st.write(f"**Word Count:** {features['word_count']}")
            st.write(f"**URL Count:** {features['url_count']}")
            st.write(f"**Scam Keywords:** {features['scam_words_count']}")
            st.write(f"**Uppercase Ratio:** {features['uppercase_ratio']:.2%}")

        # Generate explanation
        if fraud_pred == 1:
            st.subheader("🤖 AI-Generated Explanation")

            with st.spinner("Generating explanation..."):
                prompt = f"Analyze this email and explain why it's fraudulent.\n\nEmail: {email_text[:500]}\n\nExplanation:"
                inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
                outputs = llm_model.generate(**inputs, max_new_tokens=150, temperature=0.7)
                explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract just the explanation part
                if "Explanation:" in explanation:
                    explanation = explanation.split("Explanation:")[1].strip()

                st.info(explanation)

        st.success("✓ Analysis complete!")
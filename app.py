import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import pandas as pd

st.set_page_config(page_title="FT-ALBERT-QA", page_icon="🅰", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center; font-family: Block Berthold, sans serif; border-bottom: 8px solid black;border-top: 8px solid black;'>DEMO QUESTION ANSWERING WITH<br>FINE-TUNING ALBERT  MODEL</h1>", unsafe_allow_html=True)


def get_answer(question, context, model, tokenizer, max_length):
    if len(question) + len(context) + 4 > max_length:
        return None, None

    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    predicted_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    confidence_score_start = torch.nn.functional.softmax(outputs.start_logits, dim=1)[0, answer_start_index].item()
    confidence_score_end = torch.nn.functional.softmax(outputs.end_logits, dim=1)[0, answer_end_index].item()

    combined_confidence_score = (confidence_score_start + confidence_score_end) / 2

    return predicted_answer, combined_confidence_score

def generate_answer(train_df, model, tokenizer):
    sample_train_df = train_df.sample(n=3, random_state=32)

    max_length = model.config.max_position_embeddings 

    for index, row in sample_train_df.iterrows():
        question = row['Question']
        context = row['Context']
        
        predicted_answer, confidence_score = None, None
        while not predicted_answer:
            predicted_answer, confidence_score = get_answer(question, context, model, tokenizer, max_length)
            if not predicted_answer:
                row = train_df.sample(n=1).iloc[0]
                question = row['Question']
                context = row['Context']

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.text_area("Question", value=question, height=150)
            st.text_area("Context", value=context, height=150)
            st.markdown("<p style='border-bottom: 8px solid black;'></p>", unsafe_allow_html=True)
        with col2:
            st.text_area("Answer", value=predicted_answer, height=200)
            st.text_area("Confidence Score", value=confidence_score, height=100)
            st.markdown("<p style='border-bottom: 8px solid black;'></p>", unsafe_allow_html=True)




def main():
    @st.cache_data()
    def load_data():
        return pd.read_csv('train_df.csv')

    train_df = load_data()

    model_selection = st.selectbox("Select model:", ["model_1", "model_2"])

    if model_selection == "model_1":
        save_path_selection = "./models/model_1"
    elif model_selection == "model_2":
        save_path_selection = "./models/model_2"
    else:
        st.error("Invalid model selection")

    model = AutoModelForQuestionAnswering.from_pretrained(save_path_selection)
    tokenizer = AutoTokenizer.from_pretrained(save_path_selection)

    if st.button('Generate Answers', use_container_width=True):
        generate_answer(train_df, model, tokenizer)
    
if __name__ == "__main__":
    main()

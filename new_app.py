import streamlit as st
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

## Load model and tokenizer
def load_model_tokenizer(model_path = "metadata/fine_tuned", tokenizer_path = "metadata/tokenft"):
#     fine_tuned_model = "metadata/fine_tuned"
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map={"": 0}, token="auth_token")
#     fine_tuned_tokenizer = "metadata/tokenft"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,token="auth_token")
    return model, tokenizer

## Generate output
def generate(patient_note, model, tokenizer):
    system_prompt= "You are a resourceful medical assistant. Please ensure your answers are unbiased. Make sure the answers are from the text provided."
    task_desc = "Extract phrases from this text which may help understand the patient's medical condition."
    pipe = pipeline(task="text-generation", model = model, tokenizer = tokenizer, max_length=600)
    result = pipe(f"<s>[INST]<<SYS>>{system_prompt}<<SYS>>Patient Note:{patient_note}{task_desc}[/INST]")
    
    # process the keywords
    s1 = result[0]['generated_text'].split("[/INST]")
    lst = s1[1].split(',')
    a  = set()
    for x in lst: a.add(x)
    return {
        'response':s1,
        'keywords': a
    }

model, tokenizer = load_model_tokenizer()

patient_note = """
Mr. Cleveland is a 17 yo M presenting with epidosidic heart racing. This started 2-3 mos ago and he has had 5-6 episodes of tachycardia during this time. It doesn't seem to be related to exercise or public speaking. During the last episode he had SOB and felt lightheaded, but did not pass out. He also had chest pressure during this episode. Denies diaphoresis, headaches, vision changes, wt changes, fatigue, hair changes, heat or cold intolerance, changes in bowel or urinary habits, and chest and abdominal pain. No PMHx. Takes friend's adderal occassionally. No allergies. No surgeries. Dad had MI at 52; mom has thyroid disease. No tobacco or drug use. Occassionaly drinks on weekends. Sexually active with girlfriend and uses condoms.
"""

# Create a form for user input
with st.form(key='patient_note_form'):
    patient_note = st.text_area("Enter patient note:", height=200)

    submit_button = st.form_submit_button(label='Submit')

# Check if the submit button was clicked
if submit_button:
    response = generate(patient_note, model, tokenizer)

    st.write("Extracted Features:")
    for feature in response["keywords"]:   
        st.write(feature)
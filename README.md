# LLIME
LLIME - Large Language model Integrated Medical keyword Extractor

# Steps to fine tune and evaluate
- 1) Install dependencies if missing any
  2) Load data from Hugging Face repo
  3) Load base model and tokenizer
  4) Finetune and save model & tokenizer checkpoints
  5) Make sure "metadata" is in the same folder as Fine-tune_and_eval.ipynb
  6) Load finetuned model and tokenizer from checkpoint and evaluate

# Steps to run streamlit app
- 1) Ensure streamlit is installed and functional
  2) Ensure new_app.py is saved in the same folder as "metadata"
  3) Open terminal and navigate to the folder with new_app.py
  4) Run the following script: `streamlit run new_app.py`
  5) Navigate to IP

# Steps to generate an output
- 1) Enter patient note as input
  2) Hit Submit
  3) Output is generated


import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
from google.colab import drive, userdata

# --- Global Variables ---
model = None
tokenizer = None

# --- Model Initialization Function ---
def initialize_model():
    """
    Checks if the model and tokenizer are already loaded into memory.
    If not, it performs the full download and setup process.
    """
    global model, tokenizer
    if model is not None and tokenizer is not None:
        print("✅ Model and tokenizer are already loaded. Skipping initialization.")
        return

    print("Initializing model for the first time...")
    MODEL_NAME = "deepseek-ai/deepseek-coder-v2-lite-instruct"
    try:
        HF_TOKEN = userdata.get('HF_TOKEN')
    except userdata.SecretNotFoundError:
        print("Secret 'HF_TOKEN' not found. Please add it to your Colab Secrets.")
        return

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading quantized model '{MODEL_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=HF_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("\n✅ Quantized DeepSeek model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}.")
        model, tokenizer = None, None

# --- Pre-processing Function ---
def clean_and_format_question(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'\s([أ-ي])\.', text, 1)
    if len(parts) < 3: return f"Question: {text}\nChoices:"
    question_body = parts[0]
    options_part = parts[1] + '.' + parts[2]
    options_part = re.sub(r'\s([أ-ي])\.', r'\n\1.', options_part)
    return f"Question: {question_body.strip()}\nChoices:\n{options_part.strip()}"

# --- Generation and Parsing Function with Retry Logic ---
def generate_answer(question, tokenizer, model):
    # --- First Attempt: Use the standard, friendly prompt ---
    messages_attempt1 = [
        {"role": "system", "content": "You are an expert AI assistant. Analyze the following multiple-choice question and determine the correct answer. In your response, clearly state the correct letter, for example: 'The correct choice is ب.'"},
        {"role": "user", "content": question},
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(messages_attempt1, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        response_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"   -> Raw Model Response (Attempt 1): '{response_text}'")
        
        match = re.search(r'([أ-ي])', response_text)
        answer = match.group(1) if match else "PARSE_FAIL"

        # --- Second Attempt: If the first one failed, try a more direct prompt ---
        if answer == "PARSE_FAIL":
            print("   -> First attempt failed. Retrying with a more direct prompt...")
            messages_attempt2 = [
                {"role": "system", "content": "You must provide only the single correct Arabic letter as the answer. Do not include any other words, explanations, or sentences. Your entire response must be only one character."},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(messages_attempt2, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Use very few new tokens because we only expect one character
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            response_text_2 = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            print(f"   -> Raw Model Response (Attempt 2): '{response_text_2}'")
            match_2 = re.search(r'([أ-ي])', response_text_2)
            answer = match_2.group(1) if match_2 else "PARSE_FAIL"
        
        return answer

    except Exception as e:
        print(f"   -> An error occurred during model inference: {e}")
        return "ERROR"

# --- Main Prediction Pipeline ---
def run_prediction_pipeline():
    if model is None or tokenizer is None:
        print("Model is not initialized. Please run the initialization first.")
        return

    try:
        drive.mount('/content/drive')
    except:
        print("Google Drive is already mounted or failed to mount.")
    INPUT_TSV = "/content/drive/MyDrive/AraHealthQA/t2t1/subtask1_questions.tsv"
    OUTPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/deepseek_v2_answers_final_retried.csv"

    try:
        df = pd.read_csv(INPUT_TSV, sep='\t', header=None, on_bad_lines='skip')
        df.columns = ['raw_question']
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{INPUT_TSV}'.")
        return

    print(f"\nStarting prediction generation for {len(df)} questions...")
    predictions = []
    start_time = time.time()
    for index, row in df.iterrows():
        raw_question_text = row['raw_question']
        if pd.isna(raw_question_text) or not str(raw_question_text).strip():
            print(f"Skipping empty line at row {index + 1}/{len(df)}.")
            predictions.append("EMPTY_ROW")
            continue
        print(f"Processing question {index + 1}/{len(df)}...")
        formatted_question = clean_and_format_question(raw_question_text)
        answer = generate_answer(formatted_question, tokenizer, model)
        print(f"   -> Generated Answer (for CSV): {answer}")
        predictions.append(answer)

    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time) / 60:.2f} minutes.")
    pd.DataFrame(predictions).to_csv(OUTPUT_CSV, header=False, index=False)
    print("\n" + "="*50 + f"\n✅ Prediction process completed.\n📄 Results saved to '{OUTPUT_CSV}'.\n" + "="*50)

# --- Execution ---
if __name__ == "__main__":
    initialize_model()
    # You can run the pipeline multiple times without reloading the model
    run_prediction_pipeline()

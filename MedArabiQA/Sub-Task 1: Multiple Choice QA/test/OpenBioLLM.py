# --- Final Script with Data Pre-processing ---

# 49%
import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
from google.colab import drive, userdata, files

# --- Model Configuration ---
MODEL_NAME = "aaditya/OpenBioLLM-Llama3-8B"

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading quantized model '{MODEL_NAME}'...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}{% endif %}"
        print("Chat template manually set.")

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pad_token_id=tokenizer.pad_token_id,
    )
    print("\n✅ Quantized OpenBioLLM-8B model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}.")
    exit()

# --- File Paths ---
try:
    drive.mount('/content/drive')
except:
    print("Google Drive is already mounted or failed to mount.")
    
INPUT_TSV = "/content/drive/MyDrive/AraHealthQA/t2t1/subtask1_questions.tsv"
OUTPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/OpenBioLLM_8B_answers.csv"


# --- NEW: Data Cleaning and Formatting Function ---
def clean_and_format_question(text):
    """
    Cleans and standardizes the format of a raw question string to make it easier for the model.
    """
    if not isinstance(text, str):
        return ""

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Separate the question from the options based on the first Arabic letter choice
    parts = re.split(r'\s([أ-ي])\.', text, 1)
    
    if len(parts) < 3:
        # If split fails, return the cleaned text as-is
        return f"Question: {text}\nChoices:"

    question_body = parts[0]
    # Reconstruct the options part
    options_part = parts[1] + '.' + parts[2]
    
    # Standardize the options formatting (add newlines)
    options_part = re.sub(r'\s([أ-ي])\.', r'\n\1.', options_part)

    # Reconstruct in the final, clean format
    formatted_question = f"Question: {question_body.strip()}\nChoices:\n{options_part.strip()}"
    return formatted_question


# --- Function to Generate Answers ---
def generate_answer(question):
    # The system prompt is now simpler as the formatting is handled beforehand
    messages = [
        {"role": "system", "content": "You are an AI assistant. Analyze the user's question and provide the single correct Arabic letter for the answer."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output_sequences = model.generate(
            **inputs,
            max_new_tokens=10, 
            eos_token_id=terminators,
            do_sample=False, 
            temperature=None, 
            top_p=None
        )
        response_text = tokenizer.decode(output_sequences[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        print(f"   -> Raw Model Response: '{response_text}'")
        
        # Use a simple regex to find the first Arabic letter in the response
        match = re.search(r'([أ-ي])', response_text)
        return match.group(1) if match else "PARSE_FAIL"
            
    except Exception as e:
        print(f"   -> An error occurred during model inference: {e}")
        return "ERROR"


# --- Main Execution ---
def main():
    try:
        # Read the raw data file
        df = pd.read_csv(INPUT_TSV, sep='\t', header=None, on_bad_lines='skip')
        df.columns = ['raw_question']
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{INPUT_TSV}'. Please check the path and ensure Drive is mounted.")
        return

    print(f"\nStarting prediction generation for {len(df)} questions...")
    predictions, total_questions = [], len(df)
    start_time = time.time()
    
    for index, row in df.iterrows():
        raw_question = row['raw_question']
        if pd.isna(raw_question) or not str(raw_question).strip():
            print(f"Skipping empty line at row {index + 1}/{total_questions}.")
            predictions.append("EMPTY_ROW")
            continue
        
        print(f"Processing question {index + 1}/{total_questions}...")
        
        # --- APPLY THE CLEANING FUNCTION ---
        formatted_question = clean_and_format_question(raw_question)
        print(f"   -> Formatted Input:\n{formatted_question}") # Optional: to see the formatted question
        
        answer = generate_answer(formatted_question)
        predictions.append(answer)
        print(f"   -> Generated Answer (for CSV): {answer}")
    
    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time) / 60:.2f} minutes.")
    
    pd.DataFrame(predictions).to_csv(OUTPUT_CSV, header=False, index=False)
    print("\n" + "="*50 + f"\n✅ Prediction process completed.\n📄 Results saved to '{OUTPUT_CSV}'.\n" + "="*50)
    
    try:
        files.download(OUTPUT_CSV)
        print(f"\n🚀 Downloading '{OUTPUT_CSV}'...")
    except (NameError, ImportError):
        print(f"\nTo download '{OUTPUT_CSV}', please use the file browser on the left.")

if __name__ == "__main__":
    main()

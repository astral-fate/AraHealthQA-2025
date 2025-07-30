# {'accuracy': 0.29}


import os
import re
import pandas as pd
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from google.colab import drive, userdata
from googletrans import Translator

# --- Global Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- Model Loading ---
# We will load both models now to be ready for the pipeline.
# This requires significant VRAM, but you mentioned no memory limit.

# --- Primary Model: Baichuan-M1-14B-Instruct ---
PRIMARY_MODEL_NAME = "baichuan-inc/Baichuan-M1-14B-Instruct"
primary_model, primary_tokenizer = None, None
try:
    print(f"\nLoading PRIMARY model: {PRIMARY_MODEL_NAME}...")
    # This model requires trusting remote code
    primary_tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_NAME, trust_remote_code=True)
    primary_model = AutoModelForCausalLM.from_pretrained(
        PRIMARY_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"✅ Primary model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load primary model. Error: {e}")
    # We can continue if the fallback model loads
    pass

# --- Fallback Model: MedGemma ---
FALLBACK_MODEL_NAME = "google/medgemma-4b-it"
fallback_model, fallback_processor = None, None
try:
    # Ensure you have a Hugging Face token added to your Colab Secrets
    HF_TOKEN = userdata.get('HF_TOKEN')
    print(f"\nLoading FALLBACK model: {FALLBACK_MODEL_NAME}...")
    fallback_processor = AutoProcessor.from_pretrained(FALLBACK_MODEL_NAME)
    fallback_model = AutoModelForImageTextToText.from_pretrained(
        FALLBACK_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"✅ Fallback model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load fallback model. You may need to accept its license. Error: {e}")
    # If both models fail to load, we must exit.
    if primary_model is None:
        exit()
    pass


# --- Generation Functions ---

def generate_with_baichuan(question):
    """Generates an answer using the Baichuan Instruct model."""
    if not primary_model: return "ERROR"
    try:
        # Baichuan Instruct uses a specific user/assistant format
        messages = [
            {"role": "user", "content": f"Answer with a single letter only. What is the correct answer to the following question?\n\n{question}"}
        ]
        # The apply_chat_template function correctly formats the prompt
        input_ids = primary_tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = primary_model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )
        
        # Decode only the newly generated part
        response = primary_tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()
        print(f"   -> Raw Baichuan Response: '{response}'")
        match = re.search(r'([A-Za-z])', response)
        return match.group(1).upper() if match else "PARSE_FAIL"
    except Exception as e:
        print(f"   -> Baichuan inference error: {e}")
        return "ERROR"

def generate_with_medgemma(question):
    """Generates an answer using the MedGemma IT model as a fallback."""
    if not fallback_model: return "ERROR"
    try:
        print("   -> Primary model failed. Trying fallback: MedGemma...")
        messages = [{"role": "user", "content": [{"type": "text", "text": f"Answer with a single letter only. What is the correct answer to the following question?\n\n{question}"}]}]
        
        inputs = fallback_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = fallback_model.generate(**inputs, max_new_tokens=5, do_sample=False)
        
        response = fallback_processor.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        print(f"   -> Raw MedGemma Response: '{response}'")
        match = re.search(r'([A-Za-z])', response)
        return match.group(1).upper() if match else "PARSE_FAIL"
    except Exception as e:
        print(f"   -> MedGemma inference error: {e}")
        return "ERROR"

# --- Main Pipeline ---
def run_prediction_pipeline():
    """Main function that translates, predicts with fallback, and maps the answer."""
    try:
        drive.mount('/content/drive')
    except Exception as e:
        print(f"Google Drive mount failed or already mounted: {e}")
        
    translator = Translator()
    OUTPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/baichuan_fallback_answers.csv"
    INPUT_TSV = "/content/drive/MyDrive/AraHealthQA/t2t1/subtask1_questions.tsv"

    try:
        df_results = pd.read_csv(OUTPUT_CSV, header=None, names=['answer'])
        print(f"Loaded existing results from '{OUTPUT_CSV}'.")
    except FileNotFoundError:
        print(f"No previous results file found at '{OUTPUT_CSV}'. Starting from scratch.")
        df_questions_temp = pd.read_csv(INPUT_TSV, sep='\t', header=None)
        df_results = pd.DataFrame(['PARSE_FAIL'] * len(df_questions_temp), columns=['answer'])

    df_questions = pd.read_csv(INPUT_TSV, sep='\t', header=None, names=['raw_question'])
    indices_to_process = df_results[df_results['answer'].isin(['PARSE_FAIL', 'ERROR'])].index
    
    if not indices_to_process.any():
        print("\n✅ No 'PARSE_FAIL' or 'ERROR' entries found. The result file is complete.")
        return

    print(f"\nFound {len(indices_to_process)} questions to process. Starting pipeline...")
    start_time = time.time()

    for index in indices_to_process:
        raw_arabic_question = df_questions.loc[index, 'raw_question']
        print(f"Processing question {index + 1}...")

        original_arabic_options = re.findall(r'([أ-ي])\.', raw_arabic_question)
        if not original_arabic_options:
            df_results.loc[index, 'answer'] = "ERROR"
            continue

        try:
            translated_obj = translator.translate(raw_arabic_question, src='ar', dest='en')
            english_question = translated_obj.text
            print(f"   -> Translated to English: {english_question[:100]}...")
        except Exception as e:
            print(f"   -> Translation failed: {e}")
            df_results.loc[index, 'answer'] = "ERROR"
            time.sleep(2)
            continue

        # --- Attempt 1: Primary Model ---
        english_answer_char = generate_with_baichuan(english_question)

        # --- Attempt 2: Fallback Model ---
        if english_answer_char in ["PARSE_FAIL", "ERROR"]:
            english_answer_char = generate_with_medgemma(english_question)
        
        # --- Map Answer Back ---
        final_arabic_answer = "PARSE_FAIL"
        if english_answer_char not in ["PARSE_FAIL", "ERROR"]:
            try:
                answer_index = ord(english_answer_char) - ord('A')
                if 0 <= answer_index < len(original_arabic_options):
                    final_arabic_answer = original_arabic_options[answer_index]
            except (IndexError, TypeError):
                 final_arabic_answer = "PARSE_FAIL"

        df_results.loc[index, 'answer'] = final_arabic_answer
        print(f"   -> Final Parsed Answer (for CSV): {final_arabic_answer}")

        df_results.to_csv(OUTPUT_CSV, header=False, index=False)
        time.sleep(1)

    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time) / 60:.2f} minutes.")
    print("\n" + "="*50 + f"\n✅ Pipeline complete. Results saved to '{OUTPUT_CSV}'.\n" + "="*50)

if __name__ == "__main__":
    if primary_model is None and fallback_model is None:
        print("❌ Both primary and fallback models failed to load. Cannot start pipeline.")
    else:
        run_prediction_pipeline()

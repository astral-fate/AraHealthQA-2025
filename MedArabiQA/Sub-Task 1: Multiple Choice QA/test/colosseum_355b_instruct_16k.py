# 47, not cot

import os
import re
import pandas as pd
import time
from openai import OpenAI
from google.colab import drive, userdata

# --- NVIDIA API Client Setup ---
# This section sets up the connection to the NVIDIA API endpoint.

try:
    # Get your NVIDIA API key from Colab Secrets (key icon 🔑 on the left)
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
except userdata.SecretNotFoundError:
    print("❌ Secret 'NVIDIA_API_KEY' not found. Please add it to your Colab Secrets before running.")
    # Exit if the key is not found
    exit()

# Initialize the OpenAI client to point to the NVIDIA API
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

MODEL_NAME = "igenius/colosseum_355b_instruct_16k"
print(f"✅ Client configured for NVIDIA API with model: {MODEL_NAME}")


# --- Pre-processing Function ---
def clean_and_format_question(text):
    """Cleans and structures the raw question text for the model."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'\s([أ-ي])\.', text, 1)
    if len(parts) < 3: return f"Question: {text}\nChoices:"
    question_body = parts[0]
    options_part = parts[1] + '.' + parts[2]
    options_part = re.sub(r'\s([أ-ي])\.', r'\n\1.', options_part)
    return f"Question: {question_body.strip()}\nChoices:\n{options_part.strip()}"


# --- Generation and Parsing Function (Adapted for NVIDIA API) ---
def generate_answer(question):
    """
    Gets an answer from the NVIDIA API model, with a retry for failed parses.
    """
    # --- First Attempt: Standard prompt ---
    messages_attempt1 = [
        {"role": "system", "content": "You are an expert AI medical assistant. Analyze the following multiple-choice question and determine the correct answer. In your response, clearly state the correct letter, for example: 'The correct choice is ب.'"},
        {"role": "user", "content": question},
    ]
    
    try:
        # --- Handling Streaming Response ---
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_attempt1,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True  # The API returns a stream of chunks
        )
        
        # We need to assemble the full response from the stream
        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content

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
            
            completion_2 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_attempt2,
                temperature=0.2,
                top_p=0.7,
                max_tokens=5, # We only expect a single character
                stream=True
            )

            response_text_2 = ""
            for chunk in completion_2:
                if chunk.choices[0].delta.content is not None:
                    response_text_2 += chunk.choices[0].delta.content

            print(f"   -> Raw Model Response (Attempt 2): '{response_text_2}'")
            match_2 = re.search(r'([أ-ي])', response_text_2)
            answer = match_2.group(1) if match_2 else "PARSE_FAIL"
        
        return answer

    except Exception as e:
        print(f"   -> An error occurred during model inference: {e}")
        return "ERROR"


# --- Main Pipeline to Run/Rerun Predictions ---
def run_prediction_pipeline():
    try:
        drive.mount('/content/drive')
    except:
        print("Google Drive is already mounted or failed to mount.")
    
    INPUT_TSV = "/content/drive/MyDrive/AraHealthQA/t2t1/subtask1_questions.tsv"
    # Using a new output file for the new model
    OUTPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/colosseum_355b_answers.csv"

    # Try to load previous results to only re-run failures for this specific model
    try:
        df_results = pd.read_csv(OUTPUT_CSV, header=None)
        df_results.columns = ['answer']
        print(f"Loaded existing results from '{OUTPUT_CSV}'.")
    except FileNotFoundError:
        print(f"No previous results file found at '{OUTPUT_CSV}'. Starting from scratch.")
        df_questions_temp = pd.read_csv(INPUT_TSV, sep='\t', header=None)
        df_results = pd.DataFrame(['PARSE_FAIL'] * len(df_questions_temp), columns=['answer'])

    df_questions = pd.read_csv(INPUT_TSV, sep='\t', header=None)
    df_questions.columns = ['raw_question']

    failed_indices = df_results[df_results['answer'] == 'PARSE_FAIL'].index
    
    if len(failed_indices) == 0:
        print("\n✅ No 'PARSE_FAIL' entries found. The result file is complete.")
        return

    print(f"\nFound {len(failed_indices)} questions to process. Starting pipeline...")
    start_time = time.time()

    for index in failed_indices:
        raw_question_text = df_questions.loc[index, 'raw_question']
        print(f"Processing question {index + 1}...")
        
        formatted_question = clean_and_format_question(raw_question_text)
        new_answer = generate_answer(formatted_question)
        
        df_results.loc[index, 'answer'] = new_answer
        print(f"   -> Generated Answer (for CSV): {new_answer}")

    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time) / 60:.2f} minutes.")
    
    df_results.to_csv(OUTPUT_CSV, header=False, index=False)
    print("\n" + "="*50 + f"\n✅ Pipeline complete. Results saved to '{OUTPUT_CSV}'.\n" + "="*50)

# --- Execution ---
if __name__ == "__main__":
    run_prediction_pipeline()

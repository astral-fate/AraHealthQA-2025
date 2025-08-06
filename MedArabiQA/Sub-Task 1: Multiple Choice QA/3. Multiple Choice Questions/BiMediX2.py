"""
# Cell 1 (NEW VERSION): Install requirements and start the server with quantization

# The installation command remains the same
!pip install "torch==2.4.0" "numpy==1.26.4" "vllm==0.6.1.post1" "transformers==4.45.2" openai rouge-score bert-score nltk -q

import os
import time

# NEW: Added the --quantization awq flag to reduce memory usage
print("Starting vLLM Server for BiMediX2 with AWQ Quantization...")
os.system("nohup vllm serve MBZUAI/BiMediX2-8B-hf --max-model-len 32000 --port 8000 --trust-remote-code --quantization awq > vllm_server.log 2>&1 &")

# Give the server a moment to start up (waiting a bit longer to be safe)
print("Waiting for 90 seconds for the server to initialize...")
time.sleep(90)
print("✅ vLLM Server for BiMediX2 should now be running in the background.")


===============================================

import os
# This command starts the server as a background process
os.system("nohup vllm serve MBZUAI/BiMediX2-8B-hf --max-model-len 32000 --port 8000 --trust-remote-code > vllm_server.log 2>&1 &")

# Give the server a moment to start up
import time
time.sleep(60) # Wait for 1 minute for the server to initialize
print("✅ vLLM Server for BiMediX2 started in the background.")
"""

# Main script for Multiple Choice Question Answering with BiMediX2

import os
import re
import pandas as pd
from openai import OpenAI
import time
import numpy as np
from sklearn.metrics import accuracy_score # Import accuracy_score for evaluation

# --- Local Server Configuration ---
client = OpenAI(
    base_url="http://localhost:8000/v1/",
    api_key="DUMMY_KEY",
)

# --- UPDATED: File paths and column names for the MCQ task ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/multiple-choice-questions.csv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/predictions_mcq.csv' # As requested
QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer'


# --- Function to Generate Answers for MCQ Task ---
def generate_answer(question):
    """
    Sends an MCQ question to the local vLLM server, prompting the model
    to return only the single correct letter.
    """
    # --- NEW: System prompt for the MCQ task ---
    system_prompt = """You are a medical exam answering machine. Your only task is to answer the following multiple-choice medical question. Read the question and the provided options (أ, ب, ج, د, ه). Your response must be ONLY the single Arabic letter corresponding to the correct answer. Do not provide any explanation, reasoning, or any other text. For example, if option 'ب' is correct, your entire response must be 'ب'."""

    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
              model="MBZUAI/BiMediX2-8B-hf",
              messages=[
                  {"role":"system", "content": system_prompt},
                  {"role":"user","content":question}
              ],
              temperature=0.0, # Set to 0 for maximum determinism in a classification task
              max_tokens=5,    # A small value is sufficient for a single letter
            )
            response_text = completion.choices[0].message.content.strip()

            # Clean the response to ensure it's just a single character
            # This will find the first Arabic letter in the response
            arabic_letters = re.findall(r'[\u0621-\u064A]', response_text)
            if arabic_letters:
                return arabic_letters[0]
            else:
                return "" # Return empty if no letter is found

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  -> An error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return f"API Error after multiple retries: {e}"
    return f"Failed to get a response after {max_retries} attempts."


# --- NEW: Function to Evaluate MCQ Accuracy ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """
    Calculates and prints the accuracy of the MCQ predictions.
    """
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)

    # Ensure both lists are of the same length
    if len(predictions) != len(ground_truths):
        print("Warning: Prediction and ground truth lists have different lengths. Evaluation might be inaccurate.")
        # Truncate to the shorter length for comparison
        min_len = min(len(predictions), len(ground_truths))
        predictions = predictions[:min_len]
        ground_truths = ground_truths[:min_len]

    accuracy = accuracy_score(ground_truths, predictions)
    correct_predictions = sum(p == g for p, g in zip(predictions, ground_truths))
    total_predictions = len(ground_truths)

    print(f"Correct Predictions: {correct_predictions} / {total_predictions}")
    print(f"📊 Accuracy: {accuracy * 100:.2f}%")
    print("="*50 + "\n✅ Evaluation Complete.\n" + "="*50)

# --- Main Execution for MCQ Task ---
def main():
    """
    Main function for the Multiple Choice Question Answering task.
    """
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found. Please upload it first.")
        return

    if QUESTION_COLUMN not in df.columns or ANSWER_COLUMN not in df.columns:
        print(f"Error: Required columns ('{QUESTION_COLUMN}', '{ANSWER_COLUMN}') not in the CSV file.")
        return

    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Found existing prediction file: '{OUTPUT_CSV}'.")
        print("Skipping generation and loading predictions from file for evaluation.")
        predictions_df = pd.read_csv(OUTPUT_CSV, header=None)
        predictions = predictions_df[0].tolist()
    else:
        print(f"'{OUTPUT_CSV}' not found. Starting prediction generation process...")
        predictions = []
        total_questions = len(df)
        for index, row in df.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Processing question {index + 1}/{total_questions} with BiMediX2 (MCQ Mode)...")
            answer_letter = generate_answer(question)
            predictions.append(answer_letter)
            print(f"  -> Generated Answer: {answer_letter}")

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(OUTPUT_CSV, header=False, index=False)
        print(f"\nSuccessfully generated predictions and saved them to '{OUTPUT_CSV}'.")
        try:
            from google.colab import files
            files.download(OUTPUT_CSV)
        except ImportError:
            print(f"To download '{OUTPUT_CSV}', see the file browser on the left.")

    # --- UPDATED: Extracting the first character from the Answer column as the ground truth ---
    # We assume the correct letter is the first character of the 'Answer' string (e.g., "د. ...")
    ground_truths = [str(ans).strip()[0] for ans in df[ANSWER_COLUMN].tolist()]

    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

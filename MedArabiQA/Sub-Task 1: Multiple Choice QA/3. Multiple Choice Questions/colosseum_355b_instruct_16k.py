# Step 1: Install all necessary libraries
!pip install openai nltk rouge-score bert-score transformers sentencepiece -q

import os
import re
import pandas as pd
from openai import OpenAI # Using the OpenAI library for the NVIDIA API
import time
from getpass import getpass
import numpy as np
from sklearn.metrics import accuracy_score

# --- NVIDIA API Configuration ---
# Your NVIDIA API key will be accessed securely from Colab's secrets
try:
    from google.colab import userdata
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
except (ImportError, KeyError):
    print("Secret 'NVIDIA_API_KEY' not found. Please add it to the Colab secrets manager.")
    NVIDIA_API_KEY = getpass('Please enter your NVIDIA API key: ')

# Initialize the OpenAI client to point to the NVIDIA API endpoint
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

# --- File paths and column names for the MCQ task ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/multiple-choice-questions.csv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/predictions_mcq_colosseum.csv' # New file name for clarity
QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer'


# --- Function to Generate Answers for MCQ Task with Colosseum ---
def generate_answer(question):
    """
    Sends an MCQ question to the NVIDIA API to be processed by Colosseum,
    prompting the model to return only the single correct letter.
    """
    system_prompt = """You are a medical exam answering machine. Your only task is to answer the following multiple-choice medical question. Read the question and the provided options (أ, ب, ج, د, ه). Your response must be ONLY the single Arabic letter corresponding to the correct answer. Do not provide any explanation, reasoning, or any other text. For example, if option 'ب' is correct, your entire response must be 'ب'."""

    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
              messages=[
                  {"role":"system", "content": system_prompt},
                  {"role":"user","content":question}
              ],
              # --- MODEL UPDATED to Colosseum ---
              model="igenius/colosseum_355b_instruct_16k",
              temperature=0.0,
              max_tokens=5,
            )
            response_text = completion.choices[0].message.content.strip()

            # Clean the response to ensure it's just a single character
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


# --- Function to Evaluate MCQ Accuracy ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """
    Calculates and prints the accuracy of the MCQ predictions.
    """
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)

    if len(predictions) != len(ground_truths):
        print("Warning: Prediction and ground truth lists have different lengths. Evaluation might be inaccurate.")
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
            print(f"Processing question {index + 1}/{total_questions} with Colosseum (MCQ Mode)...")
            answer_letter = generate_answer(question)
            predictions.append(answer_letter)
            print(f"  -> Generated Answer: {answer_letter}")

            # Conservative delay for the NVIDIA API
            if index < total_questions - 1:
                time.sleep(1)

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(OUTPUT_CSV, header=False, index=False)
        print(f"\nSuccessfully generated predictions and saved them to '{OUTPUT_CSV}'.")
        try:
            from google.colab import files
            files.download(OUTPUT_CSV)
        except ImportError:
            print(f"To download '{OUTPUT_CSV}', see the file browser on the left.")

    # Extracting the first character from the Answer column as the ground truth
    ground_truths = [str(ans).strip()[0] for ans in df[ANSWER_COLUMN].tolist()]

    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

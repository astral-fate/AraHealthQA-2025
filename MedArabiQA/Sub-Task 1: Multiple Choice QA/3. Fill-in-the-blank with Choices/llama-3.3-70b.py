import os
import re
import pandas as pd
import groq
import time
from getpass import getpass
import numpy as np
from sklearn.metrics import accuracy_score

# --- Groq API Configuration ---
# Your Groq API key will be accessed securely from Colab's secrets
try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
except (ImportError, KeyError):
    print("Secret 'GROQ_API_KEY' not found. Please add it to the Colab secrets manager.")
    GROQ_API_KEY = getpass('Please enter your Groq API key: ')

# Initialize the Groq client
client = groq.Client(api_key=GROQ_API_KEY)


# --- File paths and column names for the MCQ task ---
# NOTE: The input CSV name suggests a "fill-in-the-blank" task, but the code logic
# is designed for Multiple Choice. Ensure this is the correct file.
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/fill-in-the-blank-choices.csv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/predictions_fitb_choices.csv'

# --- UPDATED: Correct column names as per your specification ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'


# --- Function to Generate Answers for MCQ Task ---
def generate_answer(question):
    """
    Sends an MCQ question to the Groq API, prompting the model
    to return only the single correct letter.
    """
    system_prompt = """You are a medical exam answering machine. Your only task is to answer the following multiple-choice medical question. Read the question and the provided options (أ, ب, ج, د, ه). Your response must be ONLY the single Arabic letter corresponding to the correct answer. Do not provide any explanation, reasoning, or any other text. For example, if option 'ب' is correct, your entire response must be 'ب'."""

    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
              messages=[
                  {"role":"system", "content": system_prompt},
                  {"role":"user","content":question}
              ],
              model="llama3-70b-8192", # Using a standard available model
              temperature=0.0,
              max_tokens=5,
            )
            response_text = chat_completion.choices[0].message.content.strip()

            # Clean the response to ensure it's just a single Arabic letter
            arabic_letters = re.findall(r'[\u0621-\u064A]', response_text)
            if arabic_letters:
                return arabic_letters[0]
            else:
                print(f"  -> Warning: Model returned an unexpected response: '{response_text}'. Recording as empty.")
                return "" # Return empty if no letter is found

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  -> An error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  -> API Error after multiple retries: {e}")
                return "API_ERROR"
    return "FAILED_ATTEMPTS"


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

    # Filter out API errors before calculating accuracy
    valid_indices = [i for i, p in enumerate(predictions) if p not in ["API_ERROR", "FAILED_ATTEMPTS"]]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate. Check for widespread API errors.")
        return

    accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    correct_predictions = sum(p == g for p, g in zip(valid_predictions, valid_ground_truths))
    total_valid_predictions = len(valid_predictions)
    total_questions = len(ground_truths)
    api_errors = total_questions - total_valid_predictions


    print(f"Total Questions Attempted: {total_questions}")
    print(f"API Errors/Failed Attempts: {api_errors}")
    print(f"Valid Predictions to Evaluate: {total_valid_predictions}")
    print("-" * 20)
    print(f"Correct Predictions: {correct_predictions} / {total_valid_predictions}")
    print(f"📊 Accuracy (on valid responses): {accuracy * 100:.2f}%")
    print("="*50 + "\n✅ Evaluation Complete.\n" + "="*50)


# --- Main Execution for MCQ Task ---
def main():
    """
    Main function for the Multiple Choice Question Answering task.
    """
    try:
        # Use encoding='utf-8' to handle Arabic characters properly
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # Check if the corrected column names exist in the DataFrame
    if QUESTION_COLUMN not in df.columns or ANSWER_COLUMN not in df.columns:
        print(f"Error: Required columns ('{QUESTION_COLUMN}', '{ANSWER_COLUMN}') not found in the CSV.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    # Drop rows where the question or answer is missing
    df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)


    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Found existing prediction file: '{OUTPUT_CSV}'.")
        print("Skipping generation and loading predictions from file for evaluation.")
        predictions_df = pd.read_csv(OUTPUT_CSV, header=None, encoding='utf-8')
        predictions = predictions_df[0].astype(str).tolist() # Ensure predictions are read as strings
    else:
        print(f"'{OUTPUT_CSV}' not found. Starting prediction generation process...")
        predictions = []
        total_questions = len(df)
        for index, row in df.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Processing question {index + 1}/{total_questions}...")
            answer_letter = generate_answer(question)
            predictions.append(answer_letter)
            print(f"  -> Ground Truth: {str(row[ANSWER_COLUMN]).strip()[0]} | Model's Prediction: {answer_letter}")

            # Delay for Groq API rate limit (30 RPM limit)
            if index < total_questions - 1:
                time.sleep(2.1) # Sleep for slightly over 2 seconds

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(OUTPUT_CSV, header=False, index=False, encoding='utf-8')
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
     

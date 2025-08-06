# Cell 3: Main Script
import os
import re
import pandas as pd
import time
from openai import OpenAI
from google.colab import drive
from sklearn.metrics import accuracy_score # For final evaluation

# --- Client Setup ---
# This client points to the local vLLM server we started in a previous cell.
client = OpenAI(
    base_url="http://localhost:8000/v1/",
    api_key="DUMMY_KEY", # The API key is not required for local vLLM
)

# --- Generation and Parsing Function (Corrected) ---
def generate_answer(question):
    """
    Gets an answer from the vLLM model, with corrected parsing logic.
    """
    messages = [
        {"role": "system", "content": "You are an expert AI medical assistant. Analyze the following multiple-choice question and determine the correct answer. In your response, clearly state the correct letter, for example: 'The correct choice is ب.'"},
        {"role": "user", "content": question},
    ]

    try:
        completion = client.chat.completions.create(
            model="MBZUAI/BiMediX2-8B-hf",
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
        response_text = completion.choices[0].message.content
        print(f"   -> Raw Model Response: '{response_text}'")

        # --- FIXED PARSING LOGIC ---
        # Use word boundaries (\b) to find a single, isolated Arabic letter.
        # This prevents matching letters inside words like "الجواب".
        match = re.search(r'\b([أ-ي])\b', response_text)
        
        if match:
            return match.group(1)
        else:
            # If no isolated letter is found, fall back to the original, less reliable method
            # as a last resort before declaring failure.
            fallback_match = re.search(r'([أ-ي])', response_text)
            if fallback_match:
                print("   -> Warning: Could not find isolated letter, using first letter found as fallback.")
                return fallback_match.group(1)
            
            print("   -> Parse failed. Could not find any Arabic letter.")
            return "PARSE_FAIL"

    except Exception as e:
        print(f"   -> An error occurred during model inference: {e}")
        return "ERROR"

# --- Function to Evaluate MCQ Accuracy ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """Calculates and prints the accuracy of the predictions."""
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)

    error_codes = ["ERROR", "PARSE_FAIL", ""]
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes]
    
    if not valid_indices:
        print("No valid predictions to evaluate. Check for widespread inference or parsing errors.")
        total_questions = len(ground_truths)
        failed_or_empty = total_questions
        print(f"Total Questions Attempted: {total_questions}")
        print(f"Final Unanswered / Error Count: {failed_or_empty}")
        return

    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    correct_predictions = int(accuracy * len(valid_predictions))
    total_valid_predictions = len(valid_predictions)
    total_questions = len(ground_truths)
    failed_or_empty = total_questions - total_valid_predictions

    print(f"Total Questions Attempted: {total_questions}")
    print(f"Final Unanswered / Error Count: {failed_or_empty}")
    print(f"Valid Predictions to Evaluate: {total_valid_predictions}")
    print("-" * 20)
    print(f"Correct Predictions: {correct_predictions} / {total_valid_predictions}")
    print(f"📊 Accuracy (on valid responses): {accuracy * 100:.2f}%")
    print("="*50 + "\n✅ Evaluation Complete.\n" + "="*50)


# --- Main Pipeline to Run/Rerun Predictions ---
def run_prediction_pipeline():
    try:
        drive.mount('/content/drive', force_remount=True)
    except Exception as e:
        print(f"Google Drive mount failed: {e}")
        return

    # --- File paths and column names for the bias dataset ---
    INPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/data/multiple-choice-withbias.csv"
    # Using a new output file to mark the corrected version
    OUTPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/BiMediX2_mcq_bias_answers_v2.csv"
    QUESTION_COLUMN = 'Question with Bias'
    ANSWER_COLUMN = 'Answer'

    try:
        df_questions = pd.read_csv(INPUT_CSV)
        df_questions.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN], inplace=True)
        df_questions.reset_index(drop=True, inplace=True)
    except FileNotFoundError:
        print(f"❌ Error: The input file was not found at '{INPUT_CSV}'. Please check the path.")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the input CSV: {e}")
        return

    try:
        df_results = pd.read_csv(OUTPUT_CSV, header=None)
        df_results.columns = ['answer']
        print(f"Loaded {len(df_results)} existing results from '{OUTPUT_CSV}'.")
        if len(df_results) != len(df_questions):
            print("Warning: Mismatch between number of results and questions. Starting from scratch.")
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"No previous results file found or mismatch detected. Starting from scratch.")
        df_results = pd.DataFrame(['PARSE_FAIL'] * len(df_questions), columns=['answer'])

    failed_indices = df_results[df_results['answer'].isin(['PARSE_FAIL', 'ERROR'])].index

    if len(failed_indices) == 0:
        print("\n✅ No failed entries found. The result file is complete.")
    else:
        print(f"\nFound {len(failed_indices)} questions to process. Starting pipeline...")
        start_time = time.time()

        for index in failed_indices:
            question_text = df_questions.loc[index, QUESTION_COLUMN]
            print(f"Processing question {index + 1}/{len(df_questions)}...")

            new_answer = generate_answer(question_text)

            df_results.loc[index, 'answer'] = new_answer
            print(f"   -> Generated Answer (for CSV): {new_answer}")
            
            df_results.to_csv(OUTPUT_CSV, header=False, index=False)

        end_time = time.time()
        print(f"\nTotal processing time for this run: {(end_time - start_time) / 60:.2f} minutes.")
        print("\n" + "="*50 + f"\n✅ Pipeline complete. Results saved to '{OUTPUT_CSV}'.\n" + "="*50)

    predictions = df_results['answer'].tolist()
    ground_truths = [str(ans).strip()[0] if str(ans).strip() else "INVALID_TRUTH" for ans in df_questions[ANSWER_COLUMN].tolist()]
    
    evaluate_mcq_accuracy(predictions, ground_truths)


# --- Execution ---
if __name__ == "__main__":
    run_prediction_pipeline()

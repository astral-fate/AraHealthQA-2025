# Cell 2: Your main script, updated with a new prompt and cleaning for BiMediX2

import os
import re
import pandas as pd
from openai import OpenAI
import time
import numpy as np

# Evaluation library imports
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calculator

# --- Local Server Configuration ---
client = OpenAI(
    base_url="http://localhost:8000/v1/",
    api_key="DUMMY_KEY",
)

# Input and output file names
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/fill-in-the-blank-nochoices.csv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/predictions_fitb_nochoices_BiMediX2_v2.csv' # New file version

# --- Function to Generate Answers using BiMediX2 ---
def generate_answer(question):
    """
    Sends a question to the local vLLM server to be processed by BiMediX2.
    Includes a new, highly direct prompt and a post-processing cleanup step.
    """
    # --- NEW, MORE DIRECT PROMPT FOR BIMEDIX2 ---
    system_prompt = """Your task is to provide only the precise Arabic medical term(s) that complete the user's sentence. Do not repeat the user's question. Do not add any introductory phrases like 'الجواب هو' or 'الفراغ يجب أن يملأ بـ'. Your response must contain ONLY the answer. If there are multiple answers, separate them with a comma."""

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
              temperature=0.1,
              max_tokens=1024,
            )
            response_text = completion.choices[0].message.content.strip()

            # --- NEW: POST-PROCESSING CLEANUP STEP ---
            # Define a list of common unwanted phrases to remove
            phrases_to_remove = [
                "الجواب هو",
                "الفراغ يجب أن يملأ بالكلمة",
                "الفراغ يجب أن يملأ بالعبارة",
                "الفراغات في الجملة هي",
                "املأ الفراغات بالكلمات الصحيحة",
                "املأ الفراغ بالكلمات الصحيحة",
                "املأ الفراغ بالعبارة",
                "الفراغات يجب أن تكون مملؤة بالعبارة",
                "الفراغات يجب أن تكون مملؤة بالكلمات",
                # Add any other observed junk phrases here
            ]

            cleaned_text = response_text
            for phrase in phrases_to_remove:
                cleaned_text = cleaned_text.replace(phrase, "")

            # The original filter for non-Arabic characters
            arabic_only_filter = r'[^\u0600-\u06FF\u0660-\u0669,\s]'
            final_cleaned_text = re.sub(arabic_only_filter, '', cleaned_text)

            # Remove extra commas and whitespace
            final_cleaned_text = re.sub(r' ,', ',', final_cleaned_text)
            final_cleaned_text = re.sub(r', ', ',', final_cleaned_text)

            return final_cleaned_text.strip(" ,")


        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  -> An error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return f"API Error after multiple retries: {e}"
    return f"Failed to get a response after {max_retries} attempts."


# --- Function to Evaluate Predictions (Unchanged) ---
def evaluate_predictions(predictions, ground_truths):
    """
    Calculates and prints BLEU, ROUGE-L, and BERTScore for the predictions.
    """
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)
    predictions = [str(p) for p in predictions]
    ground_truths = [str(g) for g in ground_truths]
    # BLEU Score
    bleu_scores = [sentence_bleu([truth.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25)) for pred, truth in zip(predictions, ground_truths)]
    print(f"📊 Average BLEU-4 Score: {np.mean(bleu_scores):.4f}")
    # ROUGE-L Score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_f1_scores = [scorer.score(truth, pred)['rougeL'].fmeasure for pred, truth in zip(predictions, ground_truths)]
    print(f"📊 Average ROUGE-L F1 Score: {np.mean(rouge_l_f1_scores):.4f}")
    # BERTScore
    print("⏳ Calculating BERTScore (this may take a moment)...")
    P, R, F1 = bert_score_calculator(predictions, ground_truths, lang="ar", verbose=False)
    print(f"📊 Average BERTScore F1: {F1.mean():.4f}")
    print("="*50 + "\n✅ Evaluation Complete.\n" + "="*50)

# --- Main Execution (Unchanged) ---
def main():
    """
    Main function to generate/load predictions and then evaluate them.
    """
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found. Please upload it to your Colab session.")
        return

    if 'Question - Arabic' not in df.columns or 'Answer - Arabic' not in df.columns:
        print("Error: Required columns ('Question - Arabic', 'Answer - Arabic') not in the CSV file.")
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
            question = row['Question - Arabic']
            print(f"Processing question {index + 1}/{total_questions} with BiMediX2...")
            answer = generate_answer(question)
            predictions.append(answer)
            print(f"  -> Generated Answer: {answer}")

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(OUTPUT_CSV, header=False, index=False)
        print(f"\nSuccessfully generated predictions and saved them to '{OUTPUT_CSV}'.")
        try:
            from google.colab import files
            files.download(OUTPUT_CSV)
        except ImportError:
            print(f"To download '{OUTPUT_CSV}', see the file browser on the left.")

    ground_truths = df['Answer - Arabic'].tolist()
    if len(predictions) != len(ground_truths):
        print("\n⚠️ Warning: Number of predictions does not match number of ground truths.")
        print(f"Predictions: {len(predictions)}, Ground Truths: {len(ground_truths)}")
        print("Evaluation might be incorrect. Please check your prediction file.")
    evaluate_predictions(predictions, ground_truths)


if __name__ == "__main__":
    main()

# pip install -U bitsandbytes
# pip install pandas torch transformers accelerate bitsandbytes


import os
import re
import pandas as pd
import time
from getpass import getpass
from sklearn.metrics import accuracy_score
import gc

# --- Import necessary libraries for local inference ---
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
except ImportError:
    print("Please install necessary libraries: pip install transformers torch accelerate bitsandbytes")
    exit()


# --- File paths and column names ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/fill-in-the-blank-choices.csv'
# Changed output file name for the new model
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/final_result/predictions_fitb_choices_BioMistral_SLERP.csv' 

# --- Column names ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'
# --- NEW: Added a column name for our predictions to manage state ---
PREDICTED_COLUMN = 'Predicted_Answer'


# --- Chain of Thought & Few-Shot Prompting Configuration ---
SYSTEM_PROMPT = """You are an expert medical professional and a meticulous exam assistant. Your task is to solve a multiple-choice question in Arabic.
First, you will engage in a step-by-step thinking process in a <thinking> block. Analyze the medical question, evaluate each option (أ, ب, ج, د, ه), and explain your reasoning for choosing the correct answer.
Second, after your reasoning, you MUST provide the final answer on a new line in the format:
Final Answer: [The single Arabic letter of the correct option]

This two-step process is mandatory. Your entire response must be in Arabic.
"""

# --- Function to Generate Answers (No changes needed here) ---
def generate_answer(question, model, tokenizer):
    """
    Generates an answer by manually formatting the prompt for BioMistral.
    """
    try:
        # Manually create the prompt string in the Mistral instruction format
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{question} [/INST]"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        input_length = inputs.input_ids.shape[1]
        new_tokens = outputs[0, input_length:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # --- Intelligent Parsing Logic ---
        match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if match:
            return match.group(1)

        print(f"  -> 'Final Answer' format not found. Attempting to parse reasoning...")
        
        conclusive_phrases = [
            r"الخيار الصحيح هو\s*([\u0621-\u064A])", r"الإجابة الصحيحة هي\s*([\u0621-\u064A])",
            r"الاستنتاج هو أن الخيار\s*([\u0621-\u064A])", r"الخيار\s*([\u0621-\u064A])\s*هو الصحيح",
        ]
        for phrase in conclusive_phrases:
            match = re.search(phrase, response_text)
            if match:
                print(f"  -> Found answer using conclusive phrase heuristic.")
                return match.group(1)

        option_mentions = re.findall(r"الخيار\s*([\u0621-\u064A])", response_text)
        if option_mentions:
            last_option = option_mentions[-1]
            print(f"  -> Found answer using last-mentioned option heuristic: '{last_option}'")
            return last_option
        
        print(f"  -> Warning: Could not deduce answer from response: '{response_text}'. Recording as empty.")
        return ""
            
    except Exception as e:
        print(f"  -> An error occurred during model inference: {e}")
        return "INFERENCE_ERROR"


# --- Function to Evaluate MCQ Accuracy (No changes needed here) ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """
    Calculates and prints the accuracy, normalizing different forms of Alif.
    """
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)
    
    error_codes = ["INFERENCE_ERROR", "", "INVALID_TRUTH"]
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate.")
        return

    def normalize_alif(letter):
        return letter.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    normalized_predictions = [normalize_alif(str(p)) for p in valid_predictions]
    normalized_ground_truths = [normalize_alif(str(g)) for g in valid_ground_truths]

    accuracy = accuracy_score(normalized_ground_truths, normalized_predictions)
    correct_predictions = sum(p == g for p, g in zip(normalized_ground_truths, normalized_predictions))
    
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


# --- Main Execution ---
def main():
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # --- MODIFICATION: Resume Logic ---
    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Found existing output file. Loading previous predictions to resume.")
        try:
            df_preds = pd.read_csv(OUTPUT_CSV, header=None, names=[PREDICTED_COLUMN], dtype=str, encoding='utf-8')
            # Align previous predictions with the main dataframe by index
            df[PREDICTED_COLUMN] = df_preds[PREDICTED_COLUMN]
            # Fill any rows that didn't have a prediction with an empty string
            df[PREDICTED_COLUMN].fillna('', inplace=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not read previous predictions file correctly: {e}. Starting fresh.")
            df[PREDICTED_COLUMN] = '' # Initialize with empty strings
    else:
        print("ℹ️ No previous output file found. Starting a new run.")
        df[PREDICTED_COLUMN] = '' # Initialize with empty strings
    
    # --- MODIFICATION: Filter for Unanswered Questions ---
    error_values_to_retry = ["", "INFERENCE_ERROR"]
    # Identify rows where the predicted column is either empty or has an error
    unanswered_df = df[df[PREDICTED_COLUMN].isin(error_values_to_retry) | df[PREDICTED_COLUMN].isnull()]
    
    if unanswered_df.empty:
        print("\n🎉 All questions have already been answered. Nothing to do.")
        # Proceed to final evaluation with existing data
        ground_truths = [str(ans).strip()[0] if str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
        predictions = df[PREDICTED_COLUMN].tolist()
        evaluate_mcq_accuracy(predictions, ground_truths)
        return
        
    print(f"\nFound {len(df)} total questions. {len(unanswered_df)} questions need processing.")
    
    # --- Model Loading ---
    print("="*50)
    print("🚀 Initializing local model: BioMistral/BioMistral-7B-slerp")
    print("This may take a few minutes...")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. A GPU is required for this model.")
        return
    
    try:
        print("Clearing GPU cache before model loading...")
        gc.collect()
        torch.cuda.empty_cache()
        
        model_id = "BioMistral/BioMistral-7B-slerp"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
        )

    except Exception as e:
        print(f"❌ Failed to load the model. Error: {e}")
        return

    print("✅ Model loaded successfully.")
    
    # --- Prediction Generation ---
    total_to_process = len(unanswered_df)
    processed_count = 0
    start_time = time.time()

    # --- MODIFICATION: Loop only over unanswered questions ---
    for index, row in unanswered_df.iterrows():
        processed_count += 1
        print(f"Processing question {processed_count}/{total_to_process} (Overall index: {index})...")
        
        question = row[QUESTION_COLUMN]
        answer_letter = generate_answer(question, model, tokenizer)
        
        # --- MODIFICATION: Update the main DataFrame directly ---
        df.loc[index, PREDICTED_COLUMN] = answer_letter
        
        ground_truth_letter = str(row[ANSWER_COLUMN]).strip()[0] if str(row[ANSWER_COLUMN]).strip() else "N/A"
        print(f"  -> Ground Truth: {ground_truth_letter} | Model's Predicted Letter: {answer_letter}")

        # --- MODIFICATION: Intermittent saving for resilience ---
        if processed_count % 10 == 0:
            print(f"  -> Intermittently saving progress...")
            df[PREDICTED_COLUMN].to_csv(OUTPUT_CSV, header=False, index=False, encoding='utf-8')

    end_time = time.time()
    total_duration = end_time - start_time
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    print("\n" + "="*50)
    print(f"✅ Prediction generation complete for the remaining questions.")
    if total_to_process > 0:
        print(f"⏱️  Time taken for this session: {minutes} minutes and {seconds} seconds.")
    print("="*50)
    
    # --- MODIFICATION: Save the final, complete results ---
    df[PREDICTED_COLUMN].to_csv(OUTPUT_CSV, header=False, index=False, encoding='utf-8')
    print(f"\nSuccessfully saved all predictions to '{OUTPUT_CSV}'.")

    # --- Final Evaluation ---
    ground_truths = [str(ans).strip()[0] if str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
    # --- MODIFICATION: Use the updated column from the DataFrame for evaluation ---
    final_predictions = df[PREDICTED_COLUMN].tolist()
    evaluate_mcq_accuracy(final_predictions, ground_truths)


if __name__ == "__main__":
    main()

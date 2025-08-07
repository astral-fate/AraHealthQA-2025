import os
import re
import pandas as pd
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score

# --- MODIFIED: File paths and column names updated for your dataset ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/multiple-choice-questions.csv'
# Using a new output file name to reflect the new input data
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/predictions_biomistral_mcq.csv'

# --- MODIFIED: Column names updated to match your CSV file ---
QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer'
# The 'Category' column from your file is available in the dataframe but not used in this script's logic.
# --- End of Modifications ---


# --- PIPELINE 1: Reasoning Prompt (SIMPLIFIED - NO FEW-SHOT) ---
REASONING_SYSTEM_PROMPT = """You are an expert medical professional. Your task is to analyze a multiple-choice question in Arabic.
Provide a step-by-step thinking process. Analyze the medical question, evaluate each option (أ, ب, ج, د, ه), and explain your reasoning for choosing the correct answer.
Conclude your reasoning by clearly stating which option is the most likely answer. Your entire response must be in Arabic.
"""

# --- PIPELINE 2: Extraction Prompt (SIMPLIFIED - NO FEW-SHOT) ---
ARABIC_EXTRACTION_PROMPT = """راجع التحليل الطبي التالي وحدد حرف الإجابة الصحيحة فقط لا غير.

[بداية التحليل الطبي]
{reasoning_text}
[نهاية التحليل الطبي]

الحرف الصحيح:"""

# --- Letter Mapping for Robust Parsing ---
ENGLISH_TO_ARABIC_MAP = {
    'a': 'أ', 'b': 'ب', 'c': 'ج', 'd': 'د', 'h': 'ه'
}

# --- Function to Generate Answers (SIMPLIFIED - NO FEW-SHOT) ---
def generate_answer(question, model, tokenizer):
    """
    Generates an answer using a two-step process without few-shot examples
    to prevent repetitive answer biasing.
    """
    # --- PIPELINE 1: Generate Reasoning (No Few-Shot) ---
    try:
        # The message list is now much simpler, containing only the system prompt and the user's question.
        reasoning_messages = [
            {"role": "user", "content": f"{REASONING_SYSTEM_PROMPT}\n\n{question}"},
        ]

        prompt_reasoning = tokenizer.apply_chat_template(reasoning_messages, tokenize=False, add_generation_prompt=True)
        inputs_reasoning = tokenizer(prompt_reasoning, return_tensors="pt").to(model.device)

        outputs_reasoning = model.generate(
            **inputs_reasoning, max_new_tokens=768, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
        reasoning_text = tokenizer.decode(outputs_reasoning[0, inputs_reasoning.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if not reasoning_text or reasoning_text.isspace():
             print("  -> Warning: Reasoning generation resulted in empty text.")
             return ""
        # Shortened success message for cleaner logs
        # print(f"  -> Reasoning generated successfully.")

    except Exception as e:
        print(f"  -> An error occurred during Pipeline 1 (Reasoning): {e}")
        return "INFERENCE_ERROR"

    # --- PIPELINE 2: Extract Final Answer (No Few-Shot) ---
    try:
        extraction_prompt_text = ARABIC_EXTRACTION_PROMPT.format(reasoning_text=reasoning_text)
        inputs_extraction = tokenizer(extraction_prompt_text, return_tensors="pt").to(model.device)

        outputs_extraction = model.generate(
            **inputs_extraction, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )

        final_answer_text = tokenizer.decode(outputs_extraction[0, inputs_extraction.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

        # --- Stricter Parsing Logic (No Change) ---
        arabic_match = re.search(r"([أإآابجده])", final_answer_text)
        if arabic_match:
            matched_char = arabic_match.group(1)
            if matched_char in ['ا', 'إ', 'آ']:
                return 'أ'
            return matched_char

        english_match = re.search(r"([abcdh])", final_answer_text)
        if english_match:
            english_letter = english_match.group(1)
            arabic_letter = ENGLISH_TO_ARABIC_MAP.get(english_letter, "")
            # print(f"  -> Found English letter '{english_letter}', mapped to '{arabic_letter}'.")
            return arabic_letter

        # print(f"  -> Warning: No valid option found in Pipeline 2 output: '{final_answer_text}'")
        return ""

    except Exception as e:
        print(f"  -> An error occurred during Pipeline 2 (Extraction): {e}")
        return "INFERENCE_ERROR"


# --- Main execution logic and evaluation function (No changes needed below) ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """Calculates and prints the accuracy of the model's predictions."""
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)

    def normalize_alif(letter):
        return letter.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    error_codes = ["INFERENCE_ERROR", ""]
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate.")
        return

    normalized_predictions = [normalize_alif(p) for p in valid_predictions]
    normalized_ground_truths = [normalize_alif(g) for g in valid_ground_truths]

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
    print(f"📊 Accuracy (on valid responses, Alif normalized): {accuracy * 100:.2f}%")
    print("="*50 + "\n✅ Evaluation Complete.\n" + "="*50)


def main():
    """Main function to run the prediction and evaluation process."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "BioMistral/BioMistral-7B"
        print(f"🚀 Initializing model '{model_name}' on device '{device}'...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ BioMistral model and tokenizer initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize the Hugging Face model. Error: {e}")
        return

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

    if os.path.exists(OUTPUT_CSV):
        print(f"Output file '{OUTPUT_CSV}' already exists. Please remove or rename it to run a new generation.")
        return

    print(f"'{OUTPUT_CSV}' not found. Starting a full prediction run...")

    predictions = []
    total_questions = len(df)
    start_time = time.time()

    for index, row in df.iterrows():
        question = row[QUESTION_COLUMN]
        print(f"Processing question {index + 1}/{total_questions}...")
        answer_letter = generate_answer(question, model, tokenizer)
        predictions.append(answer_letter)

        ground_truth_letter = str(row[ANSWER_COLUMN]).strip()[0] if str(row[ANSWER_COLUMN]).strip() else "N/A"
        print(f"  -> Ground Truth: {ground_truth_letter} | Model's Predicted Letter: {answer_letter}")

    end_time = time.time()
    total_duration = end_time - start_time
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    print("\n" + "="*50)
    print(f"✅ Prediction generation complete.")
    print(f"⏱️  Total time taken: {minutes} minutes and {seconds} seconds.")
    print("="*50)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(OUTPUT_CSV, header=False, index=False, encoding='utf-8')
    print(f"\nSuccessfully saved predictions to '{OUTPUT_CSV}'.")

    ground_truths = [str(ans).strip()[0] if str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

# --- Step 1: Install all necessary libraries ---
# This needs to be run once to install the required packages for the model.
# !pip install -q -U transformers bitsandbytes accelerate Pillow

# --- Step 2: Import libraries ---
import os
import re
import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Step 3: Initialize the BioMistral Model ---
# This will download the model (around 14GB) the first time it's run.
# It requires a GPU runtime in Google Colab.
print("Initializing the BioMistral pipeline...")
try:
    model_name = "BioMistral/BioMistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically use the available GPU
    )
    print("✅ BioMistral pipeline initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize pipeline. Ensure you are using a GPU runtime.")
    print(f"Error: {e}")
    # Stop execution if the model can't be loaded
    raise

# --- Step 4: File Paths and Column Names ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/multiple-choice-questions.csv'
# Updated output file name for the BioMistral model with new functionality
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/mcq/predictions_biomistral_mcq_with_accuracy.csv'
QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer' # Ground truth column - THIS IS NOW REQUIRED IN YOUR INPUT CSV

# --- Step 5: Prepare Few-Shot Examples for BioMistral ---
# The message format uses a standard chat structure.
FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": """املأ الفراغات في الجملة التالية:
في حالة الانصباب الجنبي، يشير انخفاض أو غياب الرجفان اللمسي على الجانب المصاب إلى ____، والذي ينتج عن ____.
أ. تراكم السوائل؛ عزل الصوت
ب. احتباس الهواء؛ انهيار الحويصلات الهوائية
ج. التليف؛ انخفاض مرونة الرئة
د. نمو الورم؛ انسداد الشعب الهوائية"""
    },
    {
        "role": "assistant",
        "content": """**التفكير خطوة بخطوة:**
1.  **تحليل السؤال**: يسأل السؤال عن دلالة انخفاض أو غياب "الرجفان اللمسي" (tactile fremitus) في حالة "الانصباب الجنبي" (pleural effusion) وعن الآلية المسببة لذلك. الرجفان اللمسي هو الاهتزاز الذي يمكن الشعور به على جدار الصدر أثناء الكلام.
2.  **تقييم الخيارات**:
    * **أ. تراكم السوائل؛ عزل الصوت**: الانصباب الجنبي هو بالفعل تراكم للسوائل في الغشاء الجنبي. هذا السائل يعمل كعازل، مما يمنع انتقال اهتزازات الصوت من الرئة إلى جدار الصدر. هذا يتطابق تمامًا مع находة انخفاض الرجفان اللمسي.
    * **ب. احتباس الهواء؛ انهيار الحويصلات الهوائية**: هذا يصف حالة استرواح الصدر (pneumothorax) أو انخماص الرئة (atelectasis)، والتي لها موجودات فيزيائية مختلفة.
    * **ج. التليف؛ انخفاض مرونة الرئة**: التليف الرئوي (Pulmonary fibrosis) يزيد من كثافة أنسجة الرئة، مما قد يؤدي إلى زيادة الرجفان اللمسي، وليس انخفاضه.
    * **د. نمو الورم؛ انسداد الشعب الهوائية**: قد يسبب الورم انصبابًا جنبيًا، لكن السبب المباشر لانخفاض الرجفان في هذه الحالة هو السائل نفسه الذي يعزل الصوت. الخيار "أ" يصف الآلية الفيزيائية المباشرة بشكل أفضل.
3.  **الاستنتاج**: الخيار الأكثر دقة هو أن تراكم السوائل هو ما يسبب عزل الصوت، مما يؤدي إلى انخفاض الرجفان اللمسي.

Final Answer: أ"""
    }
]

def extract_and_normalize_answer(full_text):
    """
    Parses the full text from the model to find, normalize, and translate the final answer letter.
    """
    found_letter = None
    explicit_pattern = r"(?:Final Answer|الإجابة النهائية|الإجابة الصحيحة هي|الإجابة الصحيحة|الخلاصة)\s*[:：]?\s*\**\s*([A-Ea-eأ-ي])"
    match = re.search(explicit_pattern, full_text, re.IGNORECASE | re.MULTILINE)
    if match:
        found_letter = match.group(1)

    if not found_letter:
        line_pattern = r"^\s*\**([A-Ea-eأ-ي])\.\s*.+"
        lines = full_text.splitlines()
        for line in reversed(lines):
            match = re.match(line_pattern, line.strip())
            if match:
                found_letter = match.group(1)
                break

    if not found_letter:
        lines = full_text.splitlines()
        for line in reversed(lines[-3:]):
            cleaned_line = line.strip().replace('*', '')
            if len(cleaned_line) == 1 and re.match(r"^[A-Ea-eأ-ي]$", cleaned_line):
                found_letter = cleaned_line
                break

    if not found_letter:
        return "N/A"

    found_letter = found_letter.upper()
    translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
    if found_letter in translation_map:
        found_letter = translation_map[found_letter]

    if found_letter in ['ا', 'إ', 'آ', 'أ']:
        found_letter = 'أ'

    if found_letter in ['أ', 'ب', 'ج', 'د', 'ه']:
        return found_letter
    else:
        return "Parse Error"

def extract_ground_truth_letter(answer_text):
    """
    NEW: Extracts the first Arabic letter from the ground truth answer text
    to be used for accuracy calculation.
    """
    if not isinstance(answer_text, str):
        return "N/A"
    match = re.match(r"^\s*([أ-ي])", answer_text.strip())
    if match:
        letter = match.group(1)
        if letter in ['ا', 'إ', 'آ', 'أ']:
            return 'أ'
        return letter
    return "N/A"

def get_full_reasoning(user_prompt):
    """
    Uses the local BioMistral model to get its reasoning.
    """
    messages = FEW_SHOT_MESSAGES + [{"role": "user", "content": user_prompt}]
    print("  -> 🤖 Generating response from BioMistral...")
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  -> Raw response: {response_text[:100]}...") # Print a snippet
        return response_text
    except Exception as e:
        error_message = f"An error occurred during model inference: {e}"
        print(f"\n  -> {error_message}")
        return error_message

def main():
    """
    Main function to process the CSV file with BioMistral, including resumability,
    re-attempts for failed answers, and final accuracy calculation.
    """
    output_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        full_df = pd.read_csv(INPUT_CSV)
        if QUESTION_COLUMN not in full_df.columns or ANSWER_COLUMN not in full_df.columns:
            print(f"Error: Input CSV must have '{QUESTION_COLUMN}' and '{ANSWER_COLUMN}' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Input CSV '{INPUT_CSV}' not found. Ensure Drive is mounted and the path is correct.")
        return

    # --- Resumability Logic ---
    existing_results_df = pd.DataFrame()
    if os.path.exists(OUTPUT_CSV):
        print(f"📄 Found existing results file: '{OUTPUT_CSV}'. Loading previous work.")
        existing_results_df = pd.read_csv(OUTPUT_CSV)
        processed_questions = existing_results_df[QUESTION_COLUMN].tolist()
        print(f"  -> Found {len(processed_questions)} previously processed questions.")
        df_to_process = full_df[~full_df[QUESTION_COLUMN].isin(processed_questions)].copy()
        if not df_to_process.empty:
            print(f"  -> Resuming with {len(df_to_process)} remaining questions.")
        else:
            print("  -> All questions seem to be processed. Checking for failures to re-attempt.")
    else:
        print("📄 No existing results file found. Starting from scratch.")
        df_to_process = full_df.copy()

    # --- Initial Processing of New Questions ---
    new_results_list = []
    if not df_to_process.empty:
        print("="*50)
        print(f"🚀 Starting prediction for {len(df_to_process)} new questions...")
        print("="*50)
        start_time = time.time()
        for index, row in df_to_process.iterrows():
            question = row[QUESTION_COLUMN]
            ground_truth_text = row[ANSWER_COLUMN]
            original_index = full_df.index[full_df[QUESTION_COLUMN] == question].tolist()[0]
            print(f"Processing question {original_index + 1}/{len(full_df)}: '{str(question)[:50]}...'")

            full_reasoning = get_full_reasoning(question)
            predicted_answer = extract_and_normalize_answer(full_reasoning)
            ground_truth_letter = extract_ground_truth_letter(ground_truth_text)
            print(f"  -> Ground Truth: {ground_truth_letter} | Predicted: {predicted_answer}\n")

            new_results_list.append({
                'Question': question,
                'Answer': ground_truth_text,
                'Full_Model_Reasoning': full_reasoning,
                'Ground_Truth_Letter': ground_truth_letter,
                'Final_Answer_Letter': predicted_answer
            })
        end_time = time.time()
        print(f"⏱️ New question processing time: {end_time - start_time:.2f} seconds")

    # --- Combine existing and new results ---
    new_results_df = pd.DataFrame(new_results_list)
    final_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)

    # --- Re-processing Logic for Failed Answers ("N/A" or "Parse Error") ---
    df_to_retry = final_df[final_df['Final_Answer_Letter'].isin(['N/A', 'Parse Error'])].copy()
    if not df_to_retry.empty:
        print("\n" + "="*50)
        print(f"🕵️ Found {len(df_to_retry)} questions with parsing failures. Re-attempting...")
        print("="*50)
        retry_start_time = time.time()
        for index, row in df_to_retry.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Re-processing question for index {index}: '{str(question)[:50]}...'")

            full_reasoning = get_full_reasoning(question)
            predicted_answer = extract_and_normalize_answer(full_reasoning)
            print(f"  -> Re-attempted Prediction: {predicted_answer}\n")

            # Update the DataFrame at the specific index
            final_df.loc[index, 'Full_Model_Reasoning'] = full_reasoning
            final_df.loc[index, 'Final_Answer_Letter'] = predicted_answer
        retry_end_time = time.time()
        print(f"⏱️ Re-processing time: {retry_end_time - retry_start_time:.2f} seconds")
    else:
        print("\n✅ No parsing failures found to re-attempt.")

    # --- Final Calculation and Summary ---
    if not final_df.empty:
        # Reorder columns for clarity
        cols_order = ['Question', 'Answer', 'Ground_Truth_Letter', 'Final_Answer_Letter', 'Full_Model_Reasoning']
        final_df = final_df[[col for col in cols_order if col in final_df.columns]]
        
        # Save the final, complete DataFrame
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

        # Calculate Accuracy
        valid_for_accuracy = final_df[~final_df['Ground_Truth_Letter'].isin(['N/A', 'Parse Error'])]
        correct_predictions = (valid_for_accuracy['Final_Answer_Letter'] == valid_for_accuracy['Ground_Truth_Letter']).sum()
        total_questions_for_accuracy = len(valid_for_accuracy)
        accuracy = (correct_predictions / total_questions_for_accuracy) * 100 if total_questions_for_accuracy > 0 else 0

        print("\n" + "="*50)
        print(f"✅ Processing complete.")
        print(f"📊 Final Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_questions_for_accuracy} correct)")
        print(f"💾 All results saved to '{OUTPUT_CSV}'.")
        print("="*50)


if __name__ == "__main__":
    main()

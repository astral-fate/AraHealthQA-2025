# --- Step 1: Install all necessary libraries ---
# This needs to be run once to install the required packages for the model.
# !pip install -q -U transformers bitsandbytes accelerate Pillow

# --- Step 2: Import libraries ---
import os
import re
import time
import pandas as pd
import torch
from transformers import pipeline

# --- Step 3: Initialize the MedGemma Model ---
# This will download the model (over 50GB) the first time it's run.
# It requires a GPU runtime in Google Colab.
print("Initializing the MedGemma pipeline...")
try:
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically use the available GPU
    )
    print("✅ Pipeline initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize pipeline. Ensure you are using a GPU runtime.")
    print(f"Error: {e}")
    # Stop execution if the model can't be loaded
    raise

# --- Step 4: File Paths and Column Names ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/multiple-choice-questions.csv'
# Updated output file name for the MedGemma model
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/mcq/predictions_medgemma_mcq_with_accuracy.csv'
QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer' # Ground truth column

# --- Step 5: Prepare Few-Shot Examples for MedGemma ---
FEW_SHOT_MESSAGES = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful medical assistant. Your task is to answer multiple-choice medical questions. Provide a step-by-step reasoning process and then state the final answer clearly."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": """املأ الفراغات في الجملة التالية:
في حالة الانصباب الجنبي، يشير انخفاض أو غياب الرجفان اللمسي على الجانب المصاب إلى ____، والذي ينتج عن ____.
أ. تراكم السوائل؛ عزل الصوت
ب. احتباس الهواء؛ انهيار الحويصلات الهوائية
ج. التليف؛ انخفاض مرونة الرئة
د. نمو الورم؛ انسداد الشعب الهوائية"""}]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": """**التفكير خطوة بخطوة:**
1.  **تحليل السؤال**: يسأل السؤال عن دلالة انخفاض أو غياب "الرجفان اللمسي" (tactile fremitus) في حالة "الانصباب الجنبي" (pleural effusion) وعن الآلية المسببة لذلك. الرجفان اللمسي هو الاهتزاز الذي يمكن الشعور به على جدار الصدر أثناء الكلام.
2.  **تقييم الخيارات**:
    * **أ. تراكم السوائل؛ عزل الصوت**: الانصباب الجنبي هو بالفعل تراكم للسوائل في الغشاء الجنبي. هذا السائل يعمل كعازل، مما يمنع انتقال اهتزازات الصوت من الرئة إلى جدار الصدر. هذا يتطابق تمامًا مع находة انخفاض الرجفان اللمسي.
    * **ب. احتباس الهواء؛ انهيار الحويصلات الهوائية**: هذا يصف حالة استرواح الصدر (pneumothorax) أو انخماص الرئة (atelectasis)، والتي لها موجودات فيزيائية مختلفة.
    * **ج. التليف؛ انخفاض مرونة الرئة**: التليف الرئوي (Pulmonary fibrosis) يزيد من كثافة أنسجة الرئة، مما قد يؤدي إلى زيادة الرجفان اللمسي، وليس انخفاضه.
    * **د. نمو الورم؛ انسداد الشعب الهوائية**: قد يسبب الورم انصبابًا جنبيًا، لكن السبب المباشر لانخفاض الرجفان في هذه الحالة هو السائل نفسه الذي يعزل الصوت. الخيار "أ" يصف الآلية الفيزيائية المباشرة بشكل أفضل.
3.  **الاستنتاج**: الخيار الأكثر دقة هو أن تراكم السوائل هو ما يسبب عزل الصوت، مما يؤدي إلى انخفاض الرجفان اللمسي.

Final Answer: أ"""}]
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
    Extracts the first Arabic letter from the ground truth answer text.
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
    Uses the local MedGemma pipeline to get the model's reasoning.
    """
    messages = FEW_SHOT_MESSAGES + [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]
    try:
        output = pipe(messages, max_new_tokens=1024)
        generated_content = output[0]["generated_text"][-1]["content"]
        print(generated_content)
        return generated_content
    except Exception as e:
        error_message = f"An error occurred during model inference: {e}"
        print(f"\n  -> {error_message}")
        return error_message

def main():
    """
    Main function to process the CSV with MedGemma, with accuracy calculation and resumability.
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
        if len(df_to_process) == 0:
            print("✅ All questions have already been processed. Nothing to do.")
            accuracy = (existing_results_df['Final_Answer_Letter'] == existing_results_df['Ground_Truth_Letter']).sum() / len(existing_results_df) * 100
            print(f"📊 Final accuracy from existing file: {accuracy:.2f}%")
            return
        print(f"  -> Resuming with {len(df_to_process)} remaining questions.")
    else:
        print("📄 No existing results file found. Starting from scratch.")
        df_to_process = full_df.copy()

    print("="*50)
    print(f"🚀 Starting prediction for {len(df_to_process)} questions using MedGemma...")
    print("="*50)

    start_time = time.time()
    new_results_list = []

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
            'Full_Model_Reasoning': full_reasoning,
            'Ground_Truth_Letter': ground_truth_letter,
            'Final_Answer_Letter': predicted_answer
        })

    new_results_df = pd.DataFrame(new_results_list)
    final_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # --- Final Calculation and Summary ---
    correct_predictions = (final_df['Final_Answer_Letter'] == final_df['Ground_Truth_Letter']).sum()
    total_questions = len(final_df)
    accuracy = (correct_predictions / total_questions) * 100 if total_questions > 0 else 0
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"✅ Processing complete.")
    if not new_results_df.empty:
      print(f"⏱️ Time for this session: {total_time:.2f} seconds")
    print(f"📊 Final Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_questions} correct)")
    print(f"💾 All results saved to '{OUTPUT_CSV}'.")
    print("="*50)

if __name__ == "__main__":
    main()

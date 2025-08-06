# --- Step 1: Install all necessary libraries ---
# This needs to be run once to install the required packages for the model.
!pip install -q -U transformers bitsandbytes accelerate Pillow

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
        # The following can help reduce memory usage if you encounter CUDA errors
        # load_in_4bit=True,
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
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/mcq/predictions_medgemma_mcq.csv'
QUESTION_COLUMN = 'Question'

# --- Step 5: Prepare Few-Shot Examples for MedGemma ---
# The message format is specific to this model.
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

def get_full_reasoning(user_prompt):
    """
    Uses the local MedGemma pipeline to get the model's reasoning.
    """
    # Combine the few-shot examples with the current question
    messages = FEW_SHOT_MESSAGES + [
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    try:
        # Generate the response using the pipeline
        output = pipe(messages, max_new_tokens=1024)
        # Extract the content from the last message in the generated text
        generated_content = output[0]["generated_text"][-1]["content"]
        print(generated_content) # Print the full response for real-time viewing
        return generated_content
    except Exception as e:
        error_message = f"An error occurred during model inference: {e}"
        print(f"\n  -> {error_message}")
        return error_message

def main():
    """
    Main function to process the CSV file with the MedGemma model.
    """
    output_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        df = pd.read_csv(INPUT_CSV)
        if QUESTION_COLUMN not in df.columns:
            print(f"Error: CSV must have a '{QUESTION_COLUMN}' column.")
            return
    except FileNotFoundError:
        print(f"Error: '{INPUT_CSV}' not found. Ensure Drive is mounted.")
        return

    print("="*50)
    print(f"🚀 Starting prediction for {len(df)} questions from '{INPUT_CSV}' using MedGemma...")
    print("="*50)

    start_time = time.time()
    full_reasoning_list = []
    final_answer_list = []

    for index, row in df.iterrows():
        question = row[QUESTION_COLUMN]
        print(f"Processing question {index + 1}/{len(df)}: '{str(question)[:50]}...'")

        full_reasoning = get_full_reasoning(question)
        full_reasoning_list.append(full_reasoning)

        final_answer = extract_and_normalize_answer(full_reasoning)
        final_answer_list.append(final_answer)
        print(f"  -> Extracted and Normalized Answer: {final_answer}\n")

    results_df = pd.DataFrame({
        'Question': df[QUESTION_COLUMN],
        'Full_Model_Reasoning': full_reasoning_list,
        'Final_Answer_Letter': final_answer_list
    })

    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*50)
    print(f"✅ All predictions complete.")
    print(f"💾 Results saved to '{OUTPUT_CSV}'.")
    print(f"⏱️ Total time taken: {total_time:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()

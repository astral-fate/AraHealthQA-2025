import os
import re
import time
import pandas as pd
from openai import OpenAI
from getpass import getpass

# --- API Configuration for Google Colab ---
try:
    from google.colab import userdata
    api_key = userdata.get('NVIDIA_API_KEY')
    if not api_key:
        raise ValueError("Secret 'NVIDIA_API_KEY' not found or is empty.")
except (ImportError, ValueError, KeyError):
    print("Could not find 'NVIDIA_API_KEY' in Colab secrets. Please enter it manually.")
    api_key = getpass("NVIDIA API Key: ")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

# --- File Paths and Column Names ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/multiple-choice-questions.csv'
# Updated output file name for the Palmyra-Med model
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/mcq/predictions_palmyra-med_mcq.csv'
QUESTION_COLUMN = 'Question'

# --- Few-Shot Examples to guide the model's output format ---
# This format encourages the model to provide reasoning before the final answer.
FEW_SHOT_EXAMPLES = [
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
    This function is designed to be robust and handle various output formats by searching in stages.
    """
    found_letter = None

    # Stage 1: Look for explicit answer declarations (highest priority)
    explicit_pattern = r"(?:Final Answer|الإجابة النهائية|الإجابة الصحيحة هي|الإجابة الصحيحة|الخلاصة)\s*[:：]?\s*\**\s*([A-Ea-eأ-ي])"
    match = re.search(explicit_pattern, full_text, re.IGNORECASE | re.MULTILINE)
    if match:
        found_letter = match.group(1)

    # Stage 2: If no explicit declaration, look for lines starting with a letter and a dot
    if not found_letter:
        line_pattern = r"^\s*\**([A-Ea-eأ-ي])\.\s*.+"
        lines = full_text.splitlines()
        for line in reversed(lines):
            match = re.match(line_pattern, line.strip())
            if match:
                found_letter = match.group(1)
                break

    # Stage 3: Last resort - check for a single letter on a line
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
    Makes a single API call to the Palmyra-Med model and returns the entire text.
    """
    messages = FEW_SHOT_EXAMPLES + [{"role": "user", "content": user_prompt}]
    full_response = ""
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # --- API Call updated for Palmyra-Med ---
            completion = client.chat.completions.create(
              model="writer/palmyra-med-70b-32k",
              messages=messages,
              temperature=0.2,
              top_p=0.7,
              max_tokens=1024,
              stream=True
            )

            print("  -> 🤖 Streaming response...")
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    full_response += chunk.choices[0].delta.content
            print("\n")
            return full_response

        except Exception as e:
            print(f"\n  -> An error occurred (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  -> Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return f"Failed after {max_retries} attempts: {e}"
    return "Failed to get a response."

def main():
    """
    Main function to read questions, get full reasoning, extract the final answer,
    and save everything to a new CSV file in Google Drive.
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
    print(f"🚀 Starting prediction for {len(df)} questions from '{INPUT_CSV}' using Palmyra-Med...")
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
        print(f"  -> Extracted and Normalized Answer: {final_answer}")

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

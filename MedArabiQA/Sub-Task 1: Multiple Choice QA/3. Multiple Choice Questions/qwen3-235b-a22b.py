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
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/mcq/predictions_qwen_mcq.csv'
QUESTION_COLUMN = 'Question'

# --- Few-Shot Examples to guide the model's output format ---
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
    """
    # 1. Define patterns to find the answer letter
    # This looks for keywords followed by a colon (optional) and then captures a letter.
    patterns = [
        r"(?:Final Answer|الإجابة الصحيحة|الجواب النهائي)\s*[:：]?\s*([A-Ea-eأ-ي])",
        r"([A-Ea-eأ-ي])\s*[:：]?\s*(?:is the correct answer|هي الإجابة الصحيحة)",
        r"The correct answer is\s*[:：]?\s*([A-Ea-eأ-ي])"
    ]

    found_letter = None
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            found_letter = match.group(1).upper()
            break

    # If no pattern matches, take the very last character of the string if it's a valid letter
    if not found_letter and full_text:
        last_char = full_text.strip()[-1].upper()
        if last_char in "ABCDEأبجده":
             found_letter = last_char

    if not found_letter:
        return "N/A" # Return 'Not Available' if no answer is found

    # 2. Translate English letters to Arabic
    translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
    if found_letter in translation_map:
        found_letter = translation_map[found_letter]

    # 3. Normalize different forms of Alif
    if found_letter in ['ا', 'إ']:
        found_letter = 'أ'

    return found_letter


def get_full_reasoning(user_prompt):
    """
    Makes a single API call to get the model's full reasoning and returns the entire text.
    """
    messages = FEW_SHOT_EXAMPLES + [{"role": "user", "content": user_prompt}]
    full_response = ""
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
              model="qwen/qwen3-235b-a22b",
              messages=messages,
              temperature=0.2,
              top_p=0.7,
              max_tokens=8192,
              extra_body={"chat_template_kwargs": {"thinking": True}},
              stream=True
            )

            print("  -> 🤖 Streaming response...")
            for chunk in completion:
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    print(reasoning, end="", flush=True)
                    full_response += reasoning
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
    print(f"🚀 Starting prediction for {len(df)} questions from '{INPUT_CSV}'...")
    print("="*50)

    start_time = time.time()
    full_reasoning_list = []
    final_answer_list = []

    for index, row in df.iterrows():
        question = row[QUESTION_COLUMN]
        print(f"Processing question {index + 1}/{len(df)}: '{str(question)[:50]}...'")

        # Step 1: Get the full reasoning from the model
        full_reasoning = get_full_reasoning(question)
        full_reasoning_list.append(full_reasoning)

        # Step 2: Extract and normalize the final answer from the reasoning text
        final_answer = extract_and_normalize_answer(full_reasoning)
        final_answer_list.append(final_answer)
        print(f"  -> Extracted and Normalized Answer: {final_answer}")


    # Create a new DataFrame with the results
    results_df = pd.DataFrame({
        'Question': df[QUESTION_COLUMN],
        'Full_Model_Reasoning': full_reasoning_list,
        'Final_Answer_Letter': final_answer_list
    })

    # Save the results to the output CSV
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

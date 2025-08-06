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
# Updated output file name to reflect new functionality
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/mcq/predictions_mixtral_mcq_with_accuracy.csv'
QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer' # Ground truth column

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

def extract_answer_with_compromise(full_text):
    """
    NEW: A lenient fallback function to find the most likely answer letter.
    It finds the last mentioned potential answer choice in the entire text.
    This is used when the standard parser fails.
    """
    # Regex to find any of the potential answer letters, case-insensitive for English
    pattern = r"[A-Ea-eأ-ه]"
    matches = re.findall(pattern, full_text)

    if not matches:
        return "N/A"

    # Get the last matched letter as the most likely "compromise" answer
    found_letter = matches[-1]

    # Normalize the letter
    found_letter = found_letter.upper()
    translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
    if found_letter in translation_map:
        found_letter = translation_map[found_letter]

    if found_letter in ['ا', 'إ', 'آ', 'أ']:
        found_letter = 'أ'

    if found_letter in ['أ', 'ب', 'ج', 'د', 'ه']:
        return found_letter
    else:
        # This case should be rare given the regex, but as a safeguard:
        return "Parse Error"


def extract_ground_truth_letter(answer_text):
    """
    Extracts the first Arabic letter from the ground truth answer text.
    Handles formats like 'أ. text' or 'أ text'.
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
    Makes a single API call to the Mixtral model and returns the entire text.
    """
    messages = FEW_SHOT_EXAMPLES + [{"role": "user", "content": user_prompt}]
    full_response = ""
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
              model="mistralai/mixtral-8x22b-instruct-v0.1",
              messages=messages,
              temperature=0.5,
              top_p=1,
              max_tokens=1024,
              stream=True
            )
            print("  -> 🤖 Streaming response...", end="")
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
    Main function to read questions, get predictions, calculate accuracy,
    and save everything, with the ability to resume and re-process errors.
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
    df_to_process = pd.DataFrame()
    if os.path.exists(OUTPUT_CSV):
        print(f"📄 Found existing results file: '{OUTPUT_CSV}'. Loading previous work.")
        existing_results_df = pd.read_csv(OUTPUT_CSV)

        # --- MODIFIED CHECK ---
        # Now checks for 'Ground_Truth_Letter' instead of 'Answer' in the existing file.
        required_cols = [QUESTION_COLUMN, 'Ground_Truth_Letter', 'Final_Answer_Letter']
        if not all(col in existing_results_df.columns for col in required_cols):
             print("  -> Existing CSV is missing 'Question', 'Ground_Truth_Letter', or 'Final_Answer_Letter'. Starting from scratch.")
             existing_results_df = pd.DataFrame() # Reset to start fresh
             df_to_process = full_df.copy()
        else:
            processed_questions = existing_results_df[QUESTION_COLUMN].tolist()
            print(f"  -> Found {len(processed_questions)} previously processed questions.")
            df_to_process = full_df[~full_df[QUESTION_COLUMN].isin(processed_questions)].copy()

            if len(df_to_process) == 0:
                print("✅ All questions have already been processed.")
            else:
                 print(f"  -> Resuming with {len(df_to_process)} remaining questions.")
    else:
        print("📄 No existing results file found. Starting from scratch.")
        df_to_process = full_df.copy()

    # --- Initial Processing ---
    new_results_list = []
    if not df_to_process.empty:
        print("="*50)
        print(f"🚀 Starting prediction for {len(df_to_process)} questions...")
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
            
            print(f"  -> Ground Truth: {ground_truth_letter} | Predicted: {predicted_answer}")
            
            new_results_list.append({
                'Question': question,
                'Answer': ground_truth_text, # Still needed for newly created rows
                'Full_Model_Reasoning': full_reasoning,
                'Ground_Truth_Letter': ground_truth_letter,
                'Final_Answer_Letter': predicted_answer
            })
        end_time = time.time()
        print(f"⏱️ Initial processing time: {end_time - start_time:.2f} seconds")

    # --- Combine and Save Intermediate Results ---
    new_results_df = pd.DataFrame(new_results_list)
    final_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)

    # --- Re-processing Logic for Parse Errors and N/A ---
    df_to_retry = final_df[final_df['Final_Answer_Letter'].isin(['Parse Error', 'N/A'])].copy()

    if not df_to_retry.empty:
        print("\n" + "="*50)
        print(f"🕵️ Found {len(df_to_retry)} questions with parsing errors. Re-attempting...")
        print("="*50)
        retry_start_time = time.time()

        for index, row in df_to_retry.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Re-processing question for index {index}: '{str(question)[:50]}...'")

            full_reasoning = get_full_reasoning(question)
            predicted_answer = extract_and_normalize_answer(full_reasoning)
            
            if predicted_answer in ['Parse Error', 'N/A']:
                print("  -> Standard parsing failed again. Trying compromise method...")
                predicted_answer = extract_answer_with_compromise(full_reasoning)

            print(f"  -> Re-attempted Prediction: {predicted_answer}")

            # Update the DataFrame at the specific index
            final_df.loc[index, 'Full_Model_Reasoning'] = full_reasoning
            final_df.loc[index, 'Final_Answer_Letter'] = predicted_answer
        retry_end_time = time.time()
        print(f"⏱️ Re-processing time: {retry_end_time - retry_start_time:.2f} seconds")

    else:
        print("\n✅ No parsing errors found to re-attempt.")

    # --- Final Calculation and Summary ---
    if not final_df.empty:
        # If the 'Answer' column got lost for some rows during concat, merge it back.
        if 'Answer' not in final_df.columns or final_df['Answer'].isnull().any():
             # Drop 'Answer' if it exists but is incomplete, to avoid merge conflicts
             if 'Answer' in final_df.columns:
                 final_df = final_df.drop(columns=['Answer'])
             final_df = pd.merge(final_df, full_df[[QUESTION_COLUMN, ANSWER_COLUMN]], on=QUESTION_COLUMN, how='left')
        
        # Reorder columns for clarity
        cols_order = ['Question', 'Answer', 'Ground_Truth_Letter', 'Final_Answer_Letter', 'Full_Model_Reasoning']
        # Filter for columns that actually exist in the dataframe before reordering
        final_df = final_df[[col for col in cols_order if col in final_df.columns]]
        
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

        correct_predictions = (final_df['Final_Answer_Letter'] == final_df['Ground_Truth_Letter']).sum()
        total_questions = len(final_df)
        accuracy = (correct_predictions / total_questions) * 100 if total_questions > 0 else 0
        
        print("\n" + "="*50)
        print(f"✅ Processing complete.")
        print(f"📊 Final Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_questions} correct)")
        print(f"💾 All results saved to '{OUTPUT_CSV}'.")
        print("="*50)

if __name__ == "__main__":
    main()

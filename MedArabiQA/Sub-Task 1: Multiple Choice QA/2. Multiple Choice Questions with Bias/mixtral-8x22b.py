import os
import re
import pandas as pd
import time
# Import the userdata module for Google Colab secrets
from google.colab import userdata
from sklearn.metrics import accuracy_score
from openai import OpenAI # Import the OpenAI library

# --- File paths and column names (Updated for your data) ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/multiple-choice-withbias.csv'
# Updated output file to reflect the new model
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_mixtral.csv'

# --- Column names (Set for your data) ---
# We use 'Question with Bias' to test if the model can ignore it.
# 'Answer' is the ground truth for evaluation.
QUESTION_COLUMN = 'Question with Bias'
ANSWER_COLUMN = 'Answer'


# --- Chain of Thought & Few-Shot Prompting Configuration ---
# This system prompt guides the model to follow the desired reasoning-then-answer format.
SYSTEM_PROMPT = """You are an expert medical professional and a meticulous exam assistant. Your task is to solve a multiple-choice question in Arabic.
First, provide a step-by-step thinking process. Analyze the medical question, evaluate each option (أ, ب, ج, د, ه), and explain your reasoning for choosing the correct answer.
Second, after your reasoning, you MUST provide the final answer on a new line in the format:
Final Answer: [The single Arabic letter of the correct option]

This two-step process is mandatory. Your entire response must be in Arabic.
"""

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


# --- Function to Generate Answers using NVIDIA API ---
def generate_answer(question, client):
    """
    Generates an answer using the OpenAI client pointed to the NVIDIA API.
    It accumulates the streaming response and then parses it to find the final answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    # Add a final reminder to ensure the model adheres to the format.
    final_instruction = "الآن، اتبع التعليمات بدقة. ابدأ بالتفكير خطوة بخطوة ثم اختتم إجابتك بـ 'Final Answer: ' متبوعًا بالحرف الصحيح فقط."
    prompt_with_reminder = f"{question}\n\n{final_instruction}"
    messages.append({"role": "user", "content": prompt_with_reminder})

    try:
        # The API call using the OpenAI client pointed to NVIDIA's endpoint
        # MODIFIED: Updated model and parameters as requested
        completion = client.chat.completions.create(
            model="mistralai/mixtral-8x22b-instruct-v0.1",
            messages=messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

        # Accumulate the full response from the stream
        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        
        # --- Parsing Logic (remains the same) ---

        # Method 1: Check for the standard 'Final Answer:' format first.
        match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Method 2: If standard format fails, try to deduce from the reasoning text.
        print(f"  -> 'Final Answer' format not found. Attempting to parse reasoning...")

        # Heuristic 2.1: Look for explicit conclusive phrases.
        conclusive_phrases = [
            r"الخيار الصحيح هو\s*([\u0621-\u064A])",
            r"الإجابة الصحيحة هي\s*([\u0621-\u064A])",
            r"الاستنتاج هو أن الخيار\s*([\u0621-\u064A])",
            r"الخيار\s*([\u0621-\u064A])\s*هو الصحيح",
        ]
        for phrase in conclusive_phrases:
            match = re.search(phrase, response_text)
            if match:
                print(f"  -> Found answer using conclusive phrase heuristic.")
                return match.group(1)

        # Heuristic 2.2: Assume the last mentioned option is the intended answer.
        option_mentions = re.findall(r"الخيار\s*([\u0621-\u064A])", response_text)
        if option_mentions:
            last_option = option_mentions[-1]
            print(f"  -> Found answer using last-mentioned option heuristic: '{last_option}'")
            return last_option
        
        # If all parsing fails, return empty.
        print(f"  -> Warning: Could not deduce answer from response: '{response_text}'. Recording as empty.")
        return ""
            
    except Exception as e:
        print(f"  -> An error occurred during model inference: {e}")
        return "INFERENCE_ERROR"


# --- Function to Evaluate MCQ Accuracy ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """Calculates and prints the accuracy of the model's predictions."""
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)
    
    error_codes = ["INFERENCE_ERROR", ""]
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate. Check for widespread inference or parsing errors.")
        return

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


# --- Main Execution Logic ---
def main():
    """Main function to run the prediction and evaluation process."""
    # --- Initialize OpenAI Client for NVIDIA API using Colab Secrets ---
    try:
        # Make sure you have set the 'NVIDIA_API_KEY' secret in your Colab notebook
        api_key = userdata.get('NVIDIA_API_KEY')
        client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = api_key
        )
    except Exception as e:
        print(f"❌ Failed to initialize the OpenAI client. Error: {e}")
        print("Please ensure you have set the 'NVIDIA_API_KEY' in your Google Colab secrets (sidebar > 🔑).")
        return

    print("✅ NVIDIA API client initialized successfully.")

    # --- Load and Prepare Data ---
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
    
    # --- Logic to Run Predictions (Full or Rerun) ---
    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Found existing prediction file: '{OUTPUT_CSV}'.")
        predictions_df = pd.read_csv(OUTPUT_CSV, header=None, encoding='utf-8', na_filter=False)
        predictions = predictions_df[0].astype(str).tolist()

        error_codes_to_rerun = ["INFERENCE_ERROR", ""]
        failed_indices = [i for i, p in enumerate(predictions) if p in error_codes_to_rerun]

        if not failed_indices:
            print("✅ No failed questions found to rerun. Proceeding directly to evaluation.")
        else:
            print(f"⚠️ Found {len(failed_indices)} failed questions. Starting rerun process...")
            for index in failed_indices:
                question = df.loc[index, QUESTION_COLUMN]
                print(f"Rerunning question {index + 1}/{len(df)}...")
                new_answer = generate_answer(question, client)
                predictions[index] = new_answer
                
                ground_truth_letter = str(df.loc[index, ANSWER_COLUMN]).strip()[0]
                print(f"  -> Ground Truth: {ground_truth_letter} | New Predicted Letter: {new_answer}")
            
            print("\n✅ Rerun complete. Saving updated results...")
            updated_predictions_df = pd.DataFrame(predictions)
            updated_predictions_df.to_csv(OUTPUT_CSV, header=False, index=False, encoding='utf-8')
            print(f"Successfully saved updated predictions to '{OUTPUT_CSV}'.")

    else:
        # --- This is the logic for a full run from scratch ---
        print(f"'{OUTPUT_CSV}' not found. Starting a full prediction run...")
        
        predictions = []
        total_questions = len(df)
        start_time = time.time()

        for index, row in df.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Processing question {index + 1}/{total_questions}...")
            answer_letter = generate_answer(question, client)
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

    # --- Final Evaluation ---
    ground_truths = [str(ans).strip()[0] if str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

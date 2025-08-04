import os
import re
import pandas as pd
import time
from getpass import getpass
from sklearn.metrics import accuracy_score
from openai import OpenAI # Import the OpenAI library

# --- File paths and column names (Update these paths as needed) ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/fill-in-the-blank-choices.csv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/predictions_fitb_choices_NVIDIA_qwen.csv'

# --- Column names ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'


# --- Chain of Thought & Few-Shot Prompting Configuration ---
# This system prompt and the examples guide the model to follow the desired reasoning-then-answer format.
SYSTEM_PROMPT = """You are an expert medical professional and a meticulous exam assistant. Your task is to solve a multiple-choice question in Arabic.
First, you will engage in a step-by-step thinking process in a <thinking> block. Analyze the medical question, evaluate each option (أ, ب, ج, د, ه), and explain your reasoning for choosing the correct answer.
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
        "content": """<thinking>
        1.  **تحليل السؤال**: يسأل السؤال عن دلالة انخفاض أو غياب "الرجفان اللمسي" (tactile fremitus) في حالة "الانصباب الجنبي" (pleural effusion) وعن الآلية المسببة لذلك. الرجفان اللمسي هو الاهتزاز الذي يمكن الشعور به على جدار الصدر أثناء الكلام.
        2.  **تقييم الخيارات**:
            * **أ. تراكم السوائل؛ عزل الصوت**: الانصباب الجنبي هو بالفعل تراكم للسوائل في الغشاء الجنبي. هذا السائل يعمل كعازل، مما يمنع انتقال اهتزازات الصوت من الرئة إلى جدار الصدر. هذا يتطابق تمامًا مع находة انخفاض الرجفان اللمسي.
            * **ب. احتباس الهواء؛ انهيار الحويصلات الهوائية**: هذا يصف حالة استرواح الصدر (pneumothorax) أو انخماص الرئة (atelectasis)، والتي لها موجودات فيزيائية مختلفة.
            * **ج. التليف؛ انخفاض مرونة الرئة**: التليف الرئوي (Pulmonary fibrosis) يزيد من كثافة أنسجة الرئة، مما قد يؤدي إلى زيادة الرجفان اللمسي، وليس انخفاضه.
            * **د. نمو الورم؛ انسداد الشعب الهوائية**: قد يسبب الورم انصبابًا جنبيًا، لكن السبب المباشر لانخفاض الرجفان في هذه الحالة هو السائل نفسه الذي يعزل الصوت. الخيار "أ" يصف الآلية الفيزيائية المباشرة بشكل أفضل.
        3.  **الاستنتاج**: الخيار الأكثر دقة هو أن تراكم السوائل هو ما يسبب عزل الصوت، مما يؤدي إلى انخفاض الرجفان اللمسي.
        </thinking>
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
    final_instruction = "الآن، اتبع التعليمات بدقة. ابدأ بكتلة <thinking> ثم اختتم إجابتك بـ 'Final Answer: ' متبوعًا بالحرف الصحيح فقط."
    prompt_with_reminder = f"{question}\n\n{final_instruction}"
    messages.append({"role": "user", "content": prompt_with_reminder})

    try:
        # The API call as specified, using the provided client
        completion = client.chat.completions.create(
          model="qwen/qwen3-235b-a22b",
          messages=messages,
          temperature=0.2,
          top_p=0.7,
          max_tokens=8192,
          extra_body={"chat_template_kwargs": {"thinking":True}},
          stream=True
        )

        # Accumulate the full response from the stream
        response_text = ""
        for chunk in completion:
            # The model may yield reasoning and content separately, so we combine them.
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                response_text += reasoning
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        
        # --- Parsing Logic (remains the same as the original script) ---

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
    # --- Initialize OpenAI Client for NVIDIA API ---
    # It's recommended to set the NVIDIA_API_KEY as an environment variable.
    # If not found, it will securely prompt for the key.
    try:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            print("NVIDIA_API_KEY environment variable not found.")
            api_key = getpass("Please enter your NVIDIA API Key: ")
        
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    except Exception as e:
        print(f"❌ Failed to initialize the OpenAI client. Error: {e}")
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
                # Pass the initialized client to the function
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
            # Pass the initialized client to the function
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


"""
NVIDIA_API_KEY environment variable not found.
Please enter your NVIDIA API Key: ··········
✅ NVIDIA API client initialized successfully.
'/content/drive/MyDrive/AraHealthQA/t2t1/predictions_fitb_choices_NVIDIA_qwen.csv' not found. Starting a full prediction run...
Processing question 1/100...
  -> Ground Truth: أ | Model's Predicted Letter: د
Processing question 2/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 3/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 4/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 5/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 6/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 7/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 8/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 9/100...
  -> Ground Truth: ج | Model's Predicted Letter: أ
Processing question 10/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 11/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 12/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 13/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 14/100...
  -> Ground Truth: ب | Model's Predicted Letter: د
Processing question 15/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 16/100...
  -> Ground Truth: ب | Model's Predicted Letter: أ
Processing question 17/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 18/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 19/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 20/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 21/100...
  -> Ground Truth: ج | Model's Predicted Letter: أ
Processing question 22/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 23/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 24/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 25/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 26/100...
  -> Ground Truth: أ | Model's Predicted Letter: د
Processing question 27/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 28/100...
  -> Ground Truth: د | Model's Predicted Letter: ج
Processing question 29/100...
  -> Ground Truth: أ | Model's Predicted Letter: ب
Processing question 30/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 31/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 32/100...
  -> Ground Truth: ب | Model's Predicted Letter: د
Processing question 33/100...
  -> Ground Truth: أ | Model's Predicted Letter: ب
Processing question 34/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 35/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 36/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 37/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 38/100...
  -> Ground Truth: ج | Model's Predicted Letter: د
Processing question 39/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 40/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 41/100...
  -> Ground Truth: ب | Model's Predicted Letter: أ
Processing question 42/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 43/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 44/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 45/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 46/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 47/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 48/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 49/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 50/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 51/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 52/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 53/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 54/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 55/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 56/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 57/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 58/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 59/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 60/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 61/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 62/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 63/100...
  -> Ground Truth: أ | Model's Predicted Letter: ب
Processing question 64/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 65/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 66/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 67/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 68/100...
  -> Ground Truth: ب | Model's Predicted Letter: ج
Processing question 69/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 70/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 71/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 72/100...
  -> Ground Truth: ج | Model's Predicted Letter: أ
Processing question 73/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 74/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 75/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 76/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 77/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 78/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 79/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 80/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 81/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 82/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 83/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 84/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 85/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 86/100...
  -> Ground Truth: ب | Model's Predicted Letter: أ
Processing question 87/100...
  -> Ground Truth: إ | Model's Predicted Letter: د
Processing question 88/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 89/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 90/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 91/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 92/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 93/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 94/100...
  -> Ground Truth: ج | Model's Predicted Letter: ج
Processing question 95/100...
  -> Ground Truth: د | Model's Predicted Letter: د
Processing question 96/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 97/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 98/100...
  -> Ground Truth: أ | Model's Predicted Letter: أ
Processing question 99/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب
Processing question 100/100...
  -> Ground Truth: ب | Model's Predicted Letter: ب

==================================================
✅ Prediction generation complete.
⏱️  Total time taken: 54 minutes and 7 seconds.
==================================================

Successfully saved predictions to '/content/drive/MyDrive/AraHealthQA/t2t1/predictions_fitb_choices_NVIDIA_qwen.csv'.

==================================================
🚀 Starting Evaluation...
==================================================
Total Questions Attempted: 100
Final Unanswered / Error Count: 0
Valid Predictions to Evaluate: 100
--------------------
Correct Predictions: 83 / 100
📊 Accuracy (on valid responses): 83.00%
==================================================
✅ Evaluation Complete.
==================================================
"""

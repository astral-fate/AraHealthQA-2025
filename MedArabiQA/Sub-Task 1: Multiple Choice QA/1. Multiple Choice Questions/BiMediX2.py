import os
import re
import pandas as pd
import time
from openai import OpenAI
from sklearn.metrics import accuracy_score
import gc

# --- Configuration ---
# Set the base URL for your local vLLM server
VLLM_BASE_URL = "http://localhost:8000/v1/"
# The model name as served by vLLM
MODEL_NAME = "MBZUAI/BiMediX2-8B-hf"
# A dummy API key is required by the OpenAI client, but not used by vLLM.
API_KEY = "DUMMY_KEY"

# --- File paths and column names ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/fill-in-the-blank-choices.csv'
# Updated output file to reflect the new model
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/final_result/predictions_fitb_choices_BiMediX2.csv'

# --- Column names ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'

# --- Chain of Thought & Few-Shot Prompting Configuration ---
# This prompt structure guides the model to provide a step-by-step reasoning before the final answer.
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

# --- Function to Generate Answers via vLLM API ---
def generate_answer(question, client):
    """
    Generates an answer by calling the vLLM OpenAI-compatible API.
    Includes advanced parsing to deduce the answer from the model's reasoning if the standard format is missing.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    # Add a final reminder to ensure the model follows the instructions.
    final_instruction = "الآن، اتبع التعليمات بدقة. ابدأ بكتلة <thinking> ثم اختتم إجابتك بـ 'Final Answer: ' متبوعًا بالحرف الصحيح فقط."
    prompt_with_reminder = f"{question}\n\n{final_instruction}"
    # The user's question is text-only, so we don't need the image part of the payload.
    messages.append({"role": "user", "content": prompt_with_reminder})

    try:
        # Make the API call to the local vLLM server
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1536,  # Adjusted for reasoning and answer
            temperature=0.1,
            top_p=0.9,
            stop=["<|eot_id|>"] # Add stop token if needed for the model
        )
        response_text = completion.choices[0].message.content.strip()

        # Method 1: Check for the standard 'Final Answer:' format first.
        match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Method 2: If standard format fails, parse the reasoning text.
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
        print(f"  -> An error occurred during API call: {e}")
        return "INFERENCE_ERROR"


# --- Function to Evaluate MCQ Accuracy ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """Calculates and prints the accuracy of the predictions."""
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

    # Initialize the OpenAI client to connect to the local vLLM server.
    # The client is initialized once and reused for all API calls.
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)
        print("✅ OpenAI client initialized. Ready to connect to vLLM server.")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client. Error: {e}")
        print("Please ensure the vLLM server is running and accessible.")
        return


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
        # This is the logic for a full run from scratch.
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
    # Clean up memory before starting, although less critical without local model loading.
    gc.collect()
    main()

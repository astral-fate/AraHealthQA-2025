import os
import re
import pandas as pd
import time
from openai import OpenAI
from sklearn.metrics import accuracy_score
# Import the userdata module for Google Colab secrets
from google.colab import userdata

# --- CONFIGURATION ---
# 1. Path to the original questions CSV
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/multiple-choice-withbias.csv'
# 2. Path to the CSV file with the predictions you want to fix
PREDICTIONS_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_biomistral_noshot.csv'

# Column names from your original questions file
QUESTION_COLUMN = 'Question with Bias'
ANSWER_COLUMN = 'Answer'

# --- FALLBACK MODEL PROMPT CONFIGURATION ---
# This prompt is tailored for a powerful instruction-following model.
FALLBACK_SYSTEM_PROMPT = """You are an expert medical professional and a meticulous exam assistant. Your task is to solve a multiple-choice question in Arabic.
First, provide a step-by-step thinking process. Analyze the medical question, evaluate each option (أ, ب, ج, د, ه), and explain your reasoning for choosing the correct answer.
Second, after your reasoning, you MUST provide the final answer on a new line in the format:
Final Answer: [The single Arabic letter of the correct option]

This two-step process is mandatory. Your entire response must be in Arabic.
"""

FALLBACK_FEW_SHOT_EXAMPLES = [
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


def generate_fallback_answer(question, client):
    """
    Generates an answer using the NVIDIA API with a Gemma fallback model.
    """
    messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
    messages.extend(FALLBACK_FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": question})

    try:
        # UPDATED MODEL: Using google/gemma-7b as the fallback
        # to resolve the 404 Not Found error with previous models.
        completion = client.chat.completions.create(
            model="google/gemma-7b",
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=False  # Using non-streaming for simpler parsing
        )
        response_text = completion.choices[0].message.content

        # Strict parser for "Final Answer: [letter]"
        match = re.search(r"Final Answer:\s*([أإآابجده])", response_text, re.IGNORECASE)
        if match:
            # Normalize Alif forms to 'أ' for consistency
            matched_char = match.group(1)
            if matched_char in ['ا', 'إ', 'آ']:
                return 'أ'
            return matched_char
        
        print(f"  -> Fallback Warning: Could not parse 'Final Answer:' format. Response: '{response_text[:100]}...'")
        return ""
            
    except Exception as e:
        print(f"  -> An error occurred during fallback model inference: {e}")
        return "INFERENCE_ERROR"


def evaluate_mcq_accuracy(predictions, ground_truths):
    """Calculates and prints the final accuracy of the model's predictions."""
    print("\n" + "="*50)
    print("🚀 Final Evaluation after Rerun...")
    print("="*50)
    
    def normalize_alif(letter):
        return str(letter).replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

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
    print(f"📊 Final Accuracy (on valid responses, Alif normalized): {accuracy * 100:.2f}%")
    print("="*50 + "\n✅ Rerun and Evaluation Complete.\n" + "="*50)


def main():
    """Main function to load results, rerun failures, and re-evaluate."""
    # --- Initialize Fallback OpenAI Client for NVIDIA API ---
    try:
        # Make sure you have set the 'NVIDIA_API_KEY' secret in your Colab notebook
        api_key = userdata.get('NVIDIA_API_KEY')
        fallback_client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = api_key
        )
        print("✅ NVIDIA API client for fallback model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize the OpenAI client. Error: {e}")
        print("Please ensure you have set the 'NVIDIA_API_KEY' in your Google Colab secrets (sidebar > 🔑).")
        return

    # --- Load Data ---
    try:
        df_questions = pd.read_csv(INPUT_CSV, encoding='utf-8')
        df_predictions = pd.read_csv(PREDICTIONS_CSV, header=None, names=['prediction'], encoding='utf-8', na_filter=False)
        print(f"✅ Successfully loaded questions and predictions from '{PREDICTIONS_CSV}'.")
    except FileNotFoundError:
        print(f"Error: One of the files was not found. Please check the paths:")
        print(f"  - Questions file: '{INPUT_CSV}'")
        print(f"  - Predictions file: '{PREDICTIONS_CSV}'")
        return
    
    predictions = df_predictions['prediction'].astype(str).tolist()
    
    # --- Identify Failed Questions ---
    # Failures are empty strings, "nan", or our specific error code.
    failed_indices = [i for i, p in enumerate(predictions) if p.strip() == "" or p.lower() == 'nan' or p == "INFERENCE_ERROR"]

    if not failed_indices:
        print("✅ No failed questions found to rerun. Proceeding directly to evaluation.")
    else:
        print(f"⚠️ Found {len(failed_indices)} failed questions. Starting rerun process with fallback model...")
        start_time = time.time()

        for index in failed_indices:
            question = df_questions.loc[index, QUESTION_COLUMN]
            print(f"Rerunning question {index + 1}/{len(df_questions)}...")
            
            # Call the fallback model to get a new answer
            new_answer = generate_fallback_answer(question, fallback_client)
            predictions[index] = new_answer
            
            ground_truth_letter = str(df_questions.loc[index, ANSWER_COLUMN]).strip()[0]
            print(f"  -> Ground Truth: {ground_truth_letter} | Fallback Predicted Letter: {new_answer}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        print("\n" + "="*50)
        print(f"✅ Rerun complete.")
        print(f"⏱️  Rerun time taken: {minutes} minutes and {seconds} seconds.")
        print("="*50)

        # --- Save Updated Results ---
        updated_predictions_df = pd.DataFrame(predictions)
        updated_predictions_df.to_csv(PREDICTIONS_CSV, header=False, index=False, encoding='utf-8')
        print(f"\nSuccessfully saved updated predictions back to '{PREDICTIONS_CSV}'.")

    # --- Final Evaluation ---
    ground_truths = [str(ans).strip()[0] for ans in df_questions[ANSWER_COLUMN].tolist()]
    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

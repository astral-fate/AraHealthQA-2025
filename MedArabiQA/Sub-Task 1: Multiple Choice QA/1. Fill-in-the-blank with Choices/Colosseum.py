import os
import re
import pandas as pd
import openai
import groq # For fallback
import time
from getpass import getpass
from sklearn.metrics import accuracy_score

# --- NVIDIA API Configuration (Primary Model) ---
try:
    from google.colab import userdata
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
    print("Successfully retrieved NVIDIA_API_KEY from Colab secrets.")
except (ImportError, KeyError):
    print("Could not find 'NVIDIA_API_KEY' in Colab secrets. Please enter it manually.")
    NVIDIA_API_KEY = getpass('NVIDIA API Key: ')

# Initialize the OpenAI client for NVIDIA's API endpoint
nvidia_client = openai.OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

# --- Groq API Configuration (Fallback Model) ---
try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    print("Successfully retrieved GROQ_API_KEY from Colab secrets.")
except (ImportError, KeyError):
    print("Could not find 'GROQ_API_KEY' in Colab secrets. Please enter it manually.")
    GROQ_API_KEY = getpass('Groq API Key: ')

# Initialize the Groq client
groq_client = groq.Client(api_key=GROQ_API_KEY)


# --- File paths and column names ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/fill-in-the-blank-choices.csv'
# Changed output file name for the new model
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/final_result/predictions_fitb_choices_Colosseum_with_Fallback.csv'

# --- Column names ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'


# --- Chain of Thought & Few-Shot Prompting Configuration (for Primary Model) ---
SYSTEM_PROMPT_PRIMARY = """You are an expert medical professional and a meticulous exam assistant. Your task is to solve a multiple-choice question in Arabic.
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
    },
]


# --- Fallback Function using Groq Llama 3 ---
def generate_answer_fallback(question):
    """
    Generates a direct, single-letter answer using Groq's Llama 3 model.
    This is a high-reliability, low-complexity fallback.
    """
    system_prompt = """You are a medical exam answering machine. Your only task is to answer the following multiple-choice medical question. Read the question and the provided options (أ, ب, ج, د, ه). Your response must be ONLY the single Arabic letter corresponding to the correct answer. Do not provide any explanation or other text."""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            model="llama3-70b-8192",
            temperature=0.0,
            max_tokens=5,
        )
        response_text = chat_completion.choices[0].message.content.strip()
        arabic_letters = re.findall(r'[\u0621-\u064A]', response_text)
        if arabic_letters:
            return arabic_letters[0] # Return the first Arabic letter found
        else:
            return "FALLBACK_PARSE_ERROR"
    except Exception as e:
        print(f"  -> FALLBACK FAILED: {e}")
        return "FALLBACK_API_ERROR"


# --- Primary Function to Generate Answers (MODIFIED) ---
def generate_answer_primary(question):
    """
    Sends an MCQ question to the NVIDIA API, prompting the Colosseum model to use
    Chain of Thought. This is the primary, high-effort method.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PRIMARY}
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": question})

    max_retries = 2
    retry_delay = 10
    for attempt in range(max_retries):
        try:
            # --- MODEL AND PARAMETERS CHANGED HERE ---
            completion = nvidia_client.chat.completions.create(
                model="igenius/colosseum_355b_instruct_16k",
                messages=messages,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            )
            # --- END OF CHANGES ---
            
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            
            response_text = response_text.strip()
            match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)

            if match:
                return match.group(1)
            else:
                print(f"  -> Warning: Primary model did not provide a parsable answer.")
                return "" # Return empty to trigger fallback

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  -> Primary API error: {e}. Retrying...")
                time.sleep(retry_delay)
            else:
                print(f"  -> Primary API Error after multiple retries: {e}")
                return "API_ERROR"
    return "FAILED_ATTEMPTS"


# --- Function to Evaluate MCQ Accuracy (with Normalization) ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """
    Calculates and prints the accuracy, normalizing different forms of Alif.
    """
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)

    error_codes = ["API_ERROR", "FAILED_ATTEMPTS", "", "FALLBACK_API_ERROR", "FALLBACK_PARSE_ERROR"]
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes]
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate. Check for widespread errors.")
        return

    def normalize_alif(letter):
        """Replaces all forms of Alif with a plain Alif."""
        return letter.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    normalized_predictions = [normalize_alif(p) for p in valid_predictions]
    normalized_ground_truths = [normalize_alif(g) for g in valid_ground_truths]

    accuracy = accuracy_score(normalized_ground_truths, normalized_predictions)
    correct_predictions = sum(p == g for p, g in zip(normalized_ground_truths, normalized_predictions))
    
    total_valid_predictions = len(valid_predictions)
    total_questions = len(ground_truths)
    failed_or_empty = total_questions - total_valid_predictions

    print(f"Total Questions Attempted: {total_questions}")
    print(f"Total Unanswered (after fallback): {failed_or_empty}")
    print(f"Valid Predictions to Evaluate: {total_valid_predictions}")
    print("-" * 20)
    print(f"Correct Predictions: {correct_predictions} / {total_valid_predictions}")
    print(f"📊 Accuracy (on valid responses): {accuracy * 100:.2f}%")
    print("="*50 + "\n✅ Evaluation Complete.\n" + "="*50)


# --- Main Execution ---
def main():
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    if QUESTION_COLUMN not in df.columns or ANSWER_COLUMN not in df.columns:
        print(f"Error: Required columns ('{QUESTION_COLUMN}', '{ANSWER_COLUMN}') not found in CSV.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Found existing prediction file: '{OUTPUT_CSV}'.")
        print("Skipping generation and loading predictions for evaluation.")
        predictions_df = pd.read_csv(OUTPUT_CSV, header=None, encoding='utf-8', na_filter=False)
        predictions = predictions_df[0].astype(str).tolist()
    else:
        print(f"'{OUTPUT_CSV}' not found. Starting prediction process with Colosseum and Llama 3 Fallback...")
        predictions = []
        total_questions = len(df)
        
        start_time = time.time()

        for index, row in df.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Processing question {index + 1}/{total_questions} with Colosseum (Primary)...")
            
            answer_letter = generate_answer_primary(question)
            
            if answer_letter in ["", "API_ERROR", "FAILED_ATTEMPTS"]:
                print(f"  -> Primary failed. Triggering Llama 3 (Fallback)...")
                answer_letter = generate_answer_fallback(question)
            
            predictions.append(answer_letter)
            
            try:
                ground_truth_letter = str(row[ANSWER_COLUMN]).strip()[0]
            except IndexError:
                ground_truth_letter = "N/A"
            
            print(f"  -> Ground Truth: {ground_truth_letter} | Model's Final Prediction: {answer_letter}")

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
        try:
            from google.colab import files
            files.download(OUTPUT_CSV)
        except (ImportError, NameError):
            print(f"To download '{OUTPUT_CSV}', please use the file browser in your environment.")

    ground_truths = []
    for ans in df[ANSWER_COLUMN].tolist():
        try:
            ground_truths.append(str(ans).strip()[0])
        except IndexError:
            ground_truths.append("INVALID_TRUTH")

    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

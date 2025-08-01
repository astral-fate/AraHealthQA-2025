import os
import re
import pandas as pd
import openai # Changed from groq
import time
from getpass import getpass
from sklearn.metrics import accuracy_score

# --- NVIDIA API Configuration (MODIFIED) ---
# Your NVIDIA API key will be accessed securely.
try:
    from google.colab import userdata
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
    print("Successfully retrieved NVIDIA_API_KEY from Colab secrets.")
except (ImportError, KeyError):
    print("Could not find 'NVIDIA_API_KEY' in Colab secrets.")
    print("Please enter your NVIDIA API key manually.")
    NVIDIA_API_KEY = getpass('NVIDIA API Key: ')

# Initialize the OpenAI client to point to NVIDIA's API endpoint
client = openai.OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

# --- File paths and column names for the MCQ task ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/fill-in-the-blank-choices.csv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/final_result/predictions_fitb_choices_Mixtral.csv' # Changed output file name

# --- Column names ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'


# --- Chain of Thought & Few-Shot Prompting Configuration ---
# This part remains the same as Mixtral is also a strong instruction-following model
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
    },
    {
        "role": "user",
        "content": """املأ الفراغ في الجملة التالية:
        الحمرة هي عدوى جلدية تسببها _______، وهي تصيب عادةً _____ الوجه.
        أ. المكورات العنقودية الذهبية، البشرة
        ب. العقديات B و C، الأدمة الشبكية
        ج. العقديات A و G، الأدمة الحليمية
        د. العصيات سلبية الجرام، الأنسجة تحت الجلد"""
    },
    {
        "role": "assistant",
        "content": """<thinking>
        1.  **تحليل السؤال**: يسأل السؤال عن العامل المسبب لمرض "الحمرة" (Erysipelas) والموقع التشريحي الذي يصيبه عادة في الوجه.
        2.  **تقييم الخيارات**:
            * **أ. المكورات العنقودية الذهبية، البشرة**: المكورات العنقودية تسبب التهابات جلدية أخرى مثل القوباء أو الخراجات، والحمرة تصيب طبقة أعمق من البشرة.
            * **ب. العقديات B و C، الأدمة الشبكية**: العقديات من المجموعة B و C ليست المسبب الأكثر شيوعًا للحمرة. كما أن الحمرة تصيب الطبقة العليا من الأدمة.
            * **ج. العقديات A و G، الأدمة الحليمية**: المسبب الكلاسيكي للحمرة هو العقديات الحالة للدم من المجموعة أ (Group A Streptococcus). تصيب العدوى بشكل مميز الأدمة السطحية (الأدمة الحليمية) والأوعية اللمفاوية السطحية. هذا الخيار هو الأكثر دقة.
            * **د. العصيات سلبية الجرام، الأنسجة تحت الجلد**: هذا يصف نوعًا مختلفًا تمامًا من العدوى (مثل التهاب الهلل - cellulitis) في طبقة أعمق وبكتيريا مختلفة.
        3.  **الاستنتاج**: الحمرة تسببها بشكل أساسي العقديات من المجموعة A وتصيب الأدمة الحليمية (السطحية).
        </thinking>
        Final Answer: ج"""
    }
]


# --- Function to Generate Answers for MCQ Task (MODIFIED) ---
def generate_answer(question):
    """
    Sends an MCQ question to the NVIDIA API, prompting the Mixtral model to use
    Chain of Thought. It handles the streaming response and parses the final answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": question})

    max_retries = 3
    retry_delay = 10 # Increased delay for potentially stricter rate limits
    for attempt in range(max_retries):
        try:
            # API call is now to the new client with streaming enabled
            completion = client.chat.completions.create(
                model="mistralai/mixtral-8x22b-instruct-v0.1",
                messages=messages,
                temperature=0.5,
                top_p=1,
                max_tokens=1024,
                stream=True
            )

            # Handle the streaming response
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            
            response_text = response_text.strip()
            # Parse the fully assembled response
            match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)

            if match:
                return match.group(1)
            else:
                print(f"  -> Warning: Could not parse final answer from response: '{response_text}'. Recording as empty.")
                return ""

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  -> An error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  -> API Error after multiple retries: {e}")
                return "API_ERROR"
    return "FAILED_ATTEMPTS"


# --- Function to Evaluate MCQ Accuracy (Unchanged) ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)

    if len(predictions) != len(ground_truths):
        print("Warning: Prediction and ground truth lists have different lengths.")
        min_len = min(len(predictions), len(ground_truths))
        predictions = predictions[:min_len]
        ground_truths = ground_truths[:min_len]

    valid_indices = [i for i, p in enumerate(predictions) if p not in ["API_ERROR", "FAILED_ATTEMPTS", ""]]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate. Check for widespread API or parsing errors.")
        return

    accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    correct_predictions = sum(p == g for p, g in zip(valid_predictions, valid_ground_truths))
    total_valid_predictions = len(valid_predictions)
    total_questions = len(ground_truths)
    failed_or_empty = total_questions - total_valid_predictions

    print(f"Total Questions Attempted: {total_questions}")
    print(f"API/Parsing Errors or Failed Attempts: {failed_or_empty}")
    print(f"Valid Predictions to Evaluate: {total_valid_predictions}")
    print("-" * 20)
    print(f"Correct Predictions: {correct_predictions} / {total_valid_predictions}")
    print(f"📊 Accuracy (on valid responses): {accuracy * 100:.2f}%")
    print("="*50 + "\n✅ Evaluation Complete.\n" + "="*50)


# --- Main Execution for MCQ Task (Unchanged) ---
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
        print(f"'{OUTPUT_CSV}' not found. Starting prediction generation process with Mixtral-8x22B...")
        predictions = []
        total_questions = len(df)
        
        start_time = time.time()

        for index, row in df.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Processing question {index + 1}/{total_questions}...")
            answer_letter = generate_answer(question)
            predictions.append(answer_letter)
            
            try:
                ground_truth_letter = str(row[ANSWER_COLUMN]).strip()[0]
            except IndexError:
                ground_truth_letter = "N/A"
            
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

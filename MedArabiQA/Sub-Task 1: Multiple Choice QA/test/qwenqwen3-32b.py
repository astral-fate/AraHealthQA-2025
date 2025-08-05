# https://build.nvidia.com/qwen/qwen3-235b-a22b
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm # Used for a progress bar
import re #Imported for parsing the model's output

# --- Configuration ---
# This script will attempt to load the API key from Colab secrets first.
API_KEY = None
try:
    # Modified to read from Colab secrets as requested
    from google.colab import userdata
    API_KEY = userdata.get('NVIDIA_API_KEY')
    print("Successfully loaded NVIDIA_API_KEY from Colab secrets.")
except (ImportError, KeyError):
    print("Not in a Colab environment or key not found, trying to load from environment variable.")
    API_KEY = os.environ.get("NVIDIA_API_KEY")

# If the API key is still not found, set a placeholder to trigger the error message later
if not API_KEY:
    API_KEY = "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"

# --- File Paths ---
# MODIFIED: The script now assumes you have two separate files without headers.
# One for questions and one for answers.
# PLEASE VERIFY THE PATH TO YOUR QUESTIONS FILE.
QUESTIONS_FILE = '/content/drive/MyDrive/AraHealthQA/t2t1/data/subtask1_questions.csv' #<-- PLEASE VERIFY THIS PATH
ANSWERS_FILE = '/content/drive/MyDrive/AraHealthQA/t2t1/data/subtask1_answers.csv' #<-- PLEASE VERIFY THIS PATH
OUTPUT_FILE = 'model_predictions.csv' #<-- Name of the output file

# Specify the model you want to use
MODEL_NAME = "qwen/qwen3-235b-a22b"

# --- Few-Shot Examples ---
# These examples will be included with every prompt to guide the model.
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

# --- Initialize OpenAI Client for NVIDIA API ---
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = API_KEY
)

def get_model_answer(question: str) -> str:
    """
    Queries the language model with a question and few-shot examples,
    then parses the output to extract only the final answer letter.

    Args:
        question: The question to send to the model.

    Returns:
        The extracted answer letter as a string (e.g., 'أ'),
        or an error message if something goes wrong.
    """
    # Combine few-shot examples with the current question
    messages = FEW_SHOT_EXAMPLES + [{"role": "user", "content": question}]

    try:
        completion = client.chat.completions.create(
          model=MODEL_NAME,
          messages=messages,
          temperature=0.1,
          top_p=0.7,
          max_tokens=1024,
          stream=False
        )
        
        # --- MODIFICATION START ---
        
        # Get the message object from the response
        response_message = completion.choices[0].message

        # Check if the message content is empty (None)
        if response_message.content is None:
            finish_reason = completion.choices[0].finish_reason
            print(f"\nWarning: Model returned an empty response. Finish Reason: {finish_reason}")
            return "Empty Response" # Return a specific string to avoid a crash

        # If content exists, proceed as normal
        full_response = response_message.content.strip()
        
        # --- MODIFICATION END ---
        
        # Parse the response to find the final answer letter
        match = re.search(r"Final Answer:\s*([أ-ه])", full_response)
        if match:
            return match.group(1).strip()
        else:
            # Fallback for cases where the model doesn't follow the format
            last_part = full_response.split('</thinking>')[-1]
            letters = re.findall(r"[أ-ه]", last_part)
            if letters:
                return letters[-1] # Return the last letter found
            return "Parsing Error"

    except Exception as e:
        print(f"An error occurred while querying the model: {e}")
        return "API Error"

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting model prediction generation...")

    # A simple check for the API key to avoid running without credentials
    if API_KEY == "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC" or not API_KEY:
        print("ERROR: NVIDIA_API_KEY is not set. Please set it in Colab secrets or as an environment variable.")
    else:
        try:
            print(f"Loading questions from: {QUESTIONS_FILE}")
            questions_df = pd.read_csv(QUESTIONS_FILE, header=None)

            print(f"Loading answers from: {ANSWERS_FILE}")
            answers_df = pd.read_csv(ANSWERS_FILE, header=None)

            if len(questions_df) != len(answers_df):
                print("ERROR: The number of questions and answers do not match. Please check your files.")
            else:
                ground_truth_df = pd.DataFrame({
                    'Question': questions_df[0],
                    'Ground_Truth_Answer': answers_df[0]
                })
                print("Successfully loaded and combined questions and answers.")

                predictions = []
                # Using tqdm for a progress bar
                for index, row in tqdm(ground_truth_df.iterrows(), total=len(ground_truth_df), desc="Generating Predictions"):
                    question = row['Question']
                    
                    # Get the prediction from the model
                    model_answer = get_model_answer(question)
                    predictions.append(model_answer)
                    
                # Add the predictions as a new column
                ground_truth_df['Model_Prediction'] = predictions
                
                # Save the results to a new CSV file
                ground_truth_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
                print(f"\nPredictions saved successfully to {OUTPUT_FILE}")

                # Calculate accuracy
                correct_predictions = (ground_truth_df['Ground_Truth_Answer'] == ground_truth_df['Model_Prediction']).sum()
                total_predictions = len(ground_truth_df)
                accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

                print("\n--- Evaluation Complete ---")
                print(f"Model: {MODEL_NAME}")
                print(f"Total Questions: {total_predictions}")
                print(f"Correct Predictions: {correct_predictions}")
                print(f"Final Accuracy: {accuracy:.2f}%")


        except FileNotFoundError as e:
            print(f"ERROR: A file was not found. Please check the paths for both the questions and answers files.")
            print(f"Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

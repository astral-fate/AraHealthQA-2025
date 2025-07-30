# palmyra-med-70b / palmyra-med-70b-32k 49%
import os
import re
import pandas as pd
import time
from openai import OpenAI
from google.colab import drive, userdata

# --- NVIDIA API Client Setup ---
try:
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
except userdata.SecretNotFoundError:
    print("❌ Secret 'NVIDIA_API_KEY' not found. Please add it to your Colab Secrets before running.")
    exit()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

# --- Using the Palmyra-Med model ---
MODEL_NAME = "writer/palmyra-med-70b-32k"
print(f"✅ Client configured for NVIDIA API with model: {MODEL_NAME}")

# --- Pre-processing Function ---
def clean_and_format_question(text):
    """Cleans and structures the raw question text for the model."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'\s([أ-ي])\.', text, 1)
    if len(parts) < 3: return f"Question: {text}\nChoices:"
    question_body = parts[0]
    options_part = parts[1] + '.' + parts[2]
    options_part = re.sub(r'\s([أ-ي])\.', r'\n\1.', options_part)
    return f"Question: {question_body.strip()}\nChoices:\n{options_part.strip()}"

# --- Generation and Parsing Function (with CoT and Forced-Choice Fallback) ---
def generate_answer(question):
    """
    Gets an answer from the NVIDIA API model, using CoT first and falling back
    to a forced-choice prompt if parsing fails, to eliminate PARSE_FAIL results.
    """
    # --- Attempt 1: Chain of Thought for high-quality, reasoned answer ---
    messages_attempt1 = [
        {"role": "system", "content": """You are an expert AI medical assistant. Your task is to analyze a multiple-choice question and determine the correct answer. You must be very precise and factual. Follow these steps for your response:
1.  **Analyze the Question**: Briefly explain the core medical concept being asked.
2.  **Evaluate Each Option**: Go through each option one by one. For each, state whether it is correct or incorrect and provide a brief, factual justification.
3.  **State the Final Answer**: After your analysis, you must conclude your response with the exact phrase 'The final answer is: [letter]', where [letter] is the single Arabic letter of the correct option."""},
        # Few-shot examples are included here to guide the model...
        {"role": "user", "content": "Question: كل ما هو آت صحيح عن محفظة بومان ما عدا:\nChoices:\nأ. تتداخل الاستطالات القدمية مع بعضها لتشكيل شقوق الرشح\nب. تتصل المسافة المحفظية بجوفها مع لمعة الأنبوب للقريب\nج. تعطي الوريقة الحشوية استطالات ترافق تفرعات الاوعية\nد. ترسل الخلايت تلقدمية استطالات سيتوبلاسمية باتجاه المسافة المحفظية"},
        {"role": "assistant", "content": "**1. Analyze the Question**: The question asks to identify the incorrect statement about Bowman's capsule...\n**2. Evaluate Each Option**: ...\n**3. State the Final Answer**: The final answer is: د"},
        {"role": "user", "content": "Question: كل ما يلي صحيح عن السفلس ماعدا:\nChoices:\nأ. يتميز الطور الاول بسلبية الاختبارات المصلية\nب. يتميز الطور الاول بقرحة صلبة مؤلمة غلى الاعضاء التناسلية\nج. يتميز الطور الثاني باندفاعات على الجلد والاغشية المخاطية\nد. %25 من الاجنة تموت بعد الوالدة من ام مصابة\nه. يعاني الطفل ألمصاب بالزهري الخلقي من اسنان هوتشنسن"},
        {"role": "assistant", "content": "**1. Analyze the Question**: The question asks to identify the incorrect statement about Syphilis...\n**2. Evaluate Each Option**: ...\n**3. State the Final Answer**: The final answer is: ب"},
        # Current question
        {"role": "user", "content": question},
    ]

    try:
        completion_1 = client.chat.completions.create(
            model=MODEL_NAME, messages=messages_attempt1, temperature=0.2, top_p=0.7, max_tokens=1024, stream=True
        )
        response_text_1 = "".join(chunk.choices[0].delta.content for chunk in completion_1 if chunk.choices[0].delta.content is not None)
        print(f"   -> Raw Model Response (Attempt 1 - CoT):\n---\n{response_text_1.strip()}\n---")
        
        match_1 = re.search(r'The final answer is:\s*([أ-ي])', response_text_1)
        answer = match_1.group(1) if match_1 else "PARSE_FAIL"

        # --- Attempt 2: Forced-Choice Fallback ---
        if answer == "PARSE_FAIL":
            print("   -> CoT parse failed. Falling back to forced-choice prompt...")
            messages_attempt2 = [
                {"role": "system", "content": "You must provide only the single correct Arabic letter as the answer. Do not include any other words, explanations, or sentences. Your entire response must be only one character representing the most likely correct option."},
                {"role": "user", "content": question},
            ]
            
            completion_2 = client.chat.completions.create(
                model=MODEL_NAME, messages=messages_attempt2, temperature=0.2, top_p=0.7, max_tokens=5, stream=True
            )
            response_text_2 = "".join(chunk.choices[0].delta.content for chunk in completion_2 if chunk.choices[0].delta.content is not None).strip()

            print(f"   -> Raw Model Response (Attempt 2 - Forced-Choice): '{response_text_2}'")
            # Use a more general regex since we expect just the letter
            match_2 = re.search(r'([أ-ي])', response_text_2)
            answer = match_2.group(1) if match_2 else "PARSE_FAIL"
        
        return answer

    except Exception as e:
        print(f"   -> An error occurred during model inference: {e}")
        return "ERROR"


# --- Main Pipeline to Run/Rerun Predictions ---
def run_prediction_pipeline():
    """Main function to execute the full pipeline."""
    try:
        drive.mount('/content/drive')
    except Exception as e:
        print(f"Google Drive mount failed or already mounted: {e}")

    INPUT_TSV = "/content/drive/MyDrive/AraHealthQA/t2t1/subtask1_questions.tsv"
    # --- Changed output file name to match the new model ---
    OUTPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/palmyra_med_70b_answers_CoT_fallback.csv"

    try:
        df_results = pd.read_csv(OUTPUT_CSV, header=None)
        df_results.columns = ['answer']
        print(f"Loaded existing results from '{OUTPUT_CSV}'.")
    except FileNotFoundError:
        print(f"No previous results file found at '{OUTPUT_CSV}'. Starting from scratch.")
        df_questions_temp = pd.read_csv(INPUT_TSV, sep='\t', header=None)
        df_results = pd.DataFrame(['PARSE_FAIL'] * len(df_questions_temp), columns=['answer'])

    df_questions = pd.read_csv(INPUT_TSV, sep='\t', header=None)
    df_questions.columns = ['raw_question']

    # We still use PARSE_FAIL as the trigger to process a row
    indices_to_process = df_results[df_results['answer'].isin(['PARSE_FAIL', 'ERROR'])].index
    
    if len(indices_to_process) == 0:
        print("\n✅ No 'PARSE_FAIL' or 'ERROR' entries found. The result file is complete.")
        return

    print(f"\nFound {len(indices_to_process)} questions to process. Starting pipeline...")
    start_time = time.time()

    for index in indices_to_process:
        raw_question_text = df_questions.loc[index, 'raw_question']
        print(f"Processing question {index + 1}...")
        
        formatted_question = clean_and_format_question(raw_question_text)
        new_answer = generate_answer(formatted_question)
        
        df_results.loc[index, 'answer'] = new_answer
        print(f"   -> Final Parsed Answer (for CSV): {new_answer}")

        # Save progress after each answer
        df_results.to_csv(OUTPUT_CSV, header=False, index=False)

    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time) / 60:.2f} minutes.")
    print("\n" + "="*50 + f"\n✅ Pipeline complete. Results saved to '{OUTPUT_CSV}'.\n" + "="*50)

# --- Execution ---
if __name__ == "__main__":
    run_prediction_pipeline()

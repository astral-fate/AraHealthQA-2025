import os
import re
import pandas as pd
from openai import OpenAI
import time
from google.colab import userdata, files

# --- NVIDIA API Configuration ---
# Ensure you have your NVIDIA_API_KEY set up in your Colab secrets.
try:
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
    if NVIDIA_API_KEY is None:
        raise KeyError("Secret 'NVIDIA_API_KEY' not found or is empty.")
except (ImportError, KeyError) as e:
    print(f"Error: {e}. Please ensure you have added 'NVIDIA_API_KEY' to your Colab secrets.")
    exit()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

# --- File Paths ---
# Define the input and output file paths.
INPUT_TSV  = '/content/drive/MyDrive/AraHealthQA/t2t1/subtask1_questions.tsv'
# Updated output file name to reflect the new model being used.
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/predictions_multiple_choice_mistral_7b.csv'

def generate_answer(question):
    """
    Dynamically selects a prompt based on the question type (Multiple-Choice vs. Patient Q&A),
    sends the question to the NVIDIA API using the Mistral model, and processes the response.
    """
    # This prompt is designed for multiple-choice questions.
    multiple_choice_prompt = """You are an AI expert specializing in medical and scientific questions. Your only function is to analyze the user's multiple-choice question and identify the single most accurate answer from the provided options (أ, ب, ج, د, ه, etc.).

Follow these rules perfectly:
1.  Read the question and all the options carefully. The question may ask for the correct or incorrect statement.
2.  Your output MUST be ONLY the text of the correct answer.
3.  Do NOT include the letter (e.g., 'أ.' or 'ب.'), explanations, or any other conversational text.

Example 1:
User: "نستخدم أغشية مصنوعة من ________ في تقنيات التبقيع. أ. النايلون أو السيللوز. ب. Amino acyl site. ج. Peptide site. د. الاسيتونتريل مع مادة TEAA."
Your perfect response: "النايلون أو السيللوز."

Example 2:
User: "في التهاب المشيمية نصادف الأشكال الالتهابية التالية: (الخاطئة) أ. التهاب مشيمية نتحي ب. التهاب مشيمية منتثر ج. التهاب مشيمية أمامية د. التهاب مشيمية مركزي ه. التهاب مشيمية زاوي"
Your perfect response: "التهاب مشيمية زاوي"

Now, process the user's request based on these exact rules and examples. Provide only the text of the correct answer."""

    # This prompt is a placeholder for patient Q&A questions.
    patient_qa_prompt = """You are an AI medical assistant providing triage advice...""" # Remainder of prompt omitted for brevity

    # Detect question type and select the appropriate system prompt.
    if re.search(r'أ\.|ب\.|ج\.', question):
        system_prompt = multiple_choice_prompt
        print("   -> Detected Multiple-Choice Question.")
    else:
        system_prompt = patient_qa_prompt
        print("   -> Detected Patient Q&A Question.")

    # --- API Call with Retry Logic ---
    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            # The API call now uses the specified Mistral model.
            completion = client.chat.completions.create(
              model="mistralai/mistral-7b-instruct-v0.3",
              messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
              temperature=0.1,
              top_p=0.7,
              max_tokens=100,
              stream=True
            )

            # Process the streaming response.
            raw_response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    raw_response_text += chunk.choices[0].delta.content

            print(f"   -> Raw API Response: '{raw_response_text.strip()}'")

            # Remove any <think> blocks from the response.
            text_without_think_block = re.sub(r'<think>.*?</think>', '', raw_response_text, flags=re.DOTALL)

            # Filter the text to keep only relevant characters (Arabic, English, numbers, common symbols).
            improved_filter = r'[^\u0600-\u06FF\u0660-\u06690-9a-zA-Z\s،.+-=/]'
            final_cleaned_text = re.sub(improved_filter, '', text_without_think_block)

            return final_cleaned_text.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   -> An error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"   -> API Error after multiple retries: {e}")
                return "API_ERROR"
    return "FAILED_AFTER_RETRIES"



def main():
    """
    Main function to read questions from a TSV file, generate predictions
    using the Mistral model, and save them to a CSV file.
    """
    try:
        df = pd.read_csv(INPUT_TSV, sep='\t', header=None)
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_TSV}' was not found.")
        print("Please make sure you have uploaded your 'subtask1_questions.tsv' file.")
        return

    # Updated print statement to reflect the new model.
    print(f"Starting prediction generation for {len(df)} questions using mistralai/mistral-7b-instruct-v0.3...")
    predictions = []
    total_questions = len(df)

    # Iterate through each question in the input file.
    for index, row in df.iterrows():
        question = row[0]
        if pd.isna(question) or not str(question).strip():
            print(f"Skipping empty line at row {index + 1}/{total_questions}.")
            continue

        print(f"Processing question {index + 1}/{total_questions}...")
        answer = generate_answer(str(question))
        predictions.append(answer)
        print(f"   -> Generated Answer: '{answer}'")

        # A short delay to avoid hitting API rate limits.
        if index < total_questions - 1:
            time.sleep(0.5)

    # Save the predictions to a CSV file.
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(OUTPUT_CSV, header=False, index=False)
    print("\n" + "="*50)
    print(f"✅ Successfully generated all predictions.")
    print(f"📄 Results saved to '{OUTPUT_CSV}'.")
    print("="*50)

    # Attempt to download the file in the Colab environment.
    try:
        files.download(OUTPUT_CSV)
        print(f"\n🚀 Downloading '{OUTPUT_CSV}'...")
    except (NameError, ImportError):
        print(f"\nTo download '{OUTPUT_CSV}', please use the file browser on the left.")

if __name__ == "__main__":
    main()

import os
import re
import pandas as pd
from openai import OpenAI
import time
from google.colab import userdata, files

# --- NVIDIA API Configuration ---
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
INPUT_TSV = "/content/drive/MyDrive/AraHealthQA/t2t1/subtask1_questions.tsv"
OUTPUT_CSV = "/content/drive/MyDrive/AraHealthQA/t2t1/deepseek_1_answers.csv"

# --- Function to Generate Answers ---
def generate_answer(question):
    """
    Sends a question to the NVIDIA API, asking for a chain of thought before
    the final answer. It prints the thought process and extracts only the final letter.
    """
    system_prompt = """You are an automated answering service. Your function is to first show your reasoning and then provide the correct answer. Follow these rules exactly:
1.  First, think step-by-step about the user's question. Enclose your entire reasoning process within `<thinking>` and `</thinking>` tags.
2.  After the closing `</thinking>` tag, you MUST provide the single Arabic letter (e.g., أ, ب, ت, ج, د) corresponding to the correct answer.
3.  The final letter must be the ONLY thing outside of the thinking tags.

Example:
User: "في التهاب المشيمية نصادف الأشكال الالتهابية التالية: (الخاطئة) أ. التهاب مشيمية نتحي ب. التهاب مشيمية منتثر ج. التهاب مشيمية أمامية د. التهاب مشيمية مركزي ه. التهاب مشيمية زاوي"
Your perfect response: "<thinking>The user wants the INCORRECT option. Choroiditis can be exudative, diffuse, central, and juxtapapillary (angular). Anterior choroiditis is not a standard classification. Therefore, the incorrect option is 'Anterior choroiditis'. The corresponding letter is أ.</thinking> أ"

Now, process the user's request based on these exact rules.
"""
    try:
        completion = client.chat.completions.create(
          model="igenius/colosseum_355b_instruct_16k",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": question}
          ],
          temperature=0.2,
          top_p=0.7,
          max_tokens=1024,
          stream=True
        )

        raw_response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                raw_response_text += chunk.choices[0].delta.content

        print(f"   -> Raw Model Response (with thought process):\n{raw_response_text}\n" + "-"*20)

        final_answer_text = re.sub(r'<thinking>.*?</thinking>', '', raw_response_text, flags=re.DOTALL).strip()

        # MODIFIED: Changed re.match to re.search for more flexible parsing.
        # This will now correctly find 'ه' in 'هـ' and ignore the extra character.
        match = re.search(r'([أ-ي])', final_answer_text)

        if match:
            final_answer = match.group(1)
        else:
            final_answer = "PARSE_FAIL"

        return final_answer

    except Exception as e:
        print(f"   -> An API or other error occurred: {e}")
        return "ERROR"

# --- Main Execution ---
def main():
    """
    Reads questions from a TSV file, generates predictions with thought processes,
    and saves only the final letter answers.
    """
    try:
        df = pd.read_csv(INPUT_TSV, sep='\t', header=None, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_TSV}' was not found.")
        return

    print(f"Starting prediction generation for {len(df)} questions using '{'igenius/colosseum_355b_instruct_16k'}'...")
    predictions = []
    total_questions = len(df)

    for index, row in df.iterrows():
        question = row[0]
        if pd.isna(question) or not str(question).strip():
            print(f"Skipping empty line at row {index + 1}/{total_questions}.")
            predictions.append("EMPTY_ROW")
            continue

        print(f"Processing question {index + 1}/{total_questions}...")
        answer = generate_answer(str(question).strip())
        predictions.append(answer)
        print(f"   -> Generated Answer (for CSV): {answer}")

        if index < total_questions - 1:
            time.sleep(1)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(OUTPUT_CSV, header=False, index=False)
    print("\n" + "="*50)
    print(f"✅ Prediction process completed.")
    print(f"📄 Results saved to '{OUTPUT_CSV}'. The file will contain only the final letters.")
    print("="*50)

    try:
        files.download(OUTPUT_CSV)
        print(f"\n🚀 Downloading '{OUTPUT_CSV}'...")
    except (NameError, ImportError):
        print(f"\nTo download '{OUTPUT_CSV}', please use the file browser on the left.")

if __name__ == "__main__":
    main()

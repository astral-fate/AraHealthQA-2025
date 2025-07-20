import os
import re
import pandas as pd
from openai import OpenAI
import time
from google.colab import userdata, files

# --- NVIDIA API Configuration ---
# Securely access your NVIDIA API key from Colab's secrets.
try:
    # This securely retrieves the API key you've stored in Colab's secrets.
    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')
    if NVIDIA_API_KEY is None:
        raise KeyError("Secret 'NVIDIA_API_KEY' not found or is empty.")
except (ImportError, KeyError) as e:
    print(f"Error: {e}. Please ensure you have added 'NVIDIA_API_KEY' to your Colab secrets.")
    # Exit the script if the API key is not available.
    exit()

# Initialize the OpenAI client to point to the NVIDIA API endpoint
# This uses the API key retrieved securely above.
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

INPUT_TSV  = '/content/drive/MyDrive/AraHealthQA/fitbt2/subtask2_questions.tsv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/fitbt2/predictions_fitb_nochoices_deepseek.csv' # Changed output filename for clarity

# --- Function to Generate Answers ---
def generate_answer(question):
    """
    Sends a question to the NVIDIA API to be processed by the deepseek model
    using a streaming response. It accumulates the streamed content and then
    applies cleaning logic to extract only the Arabic answer.
    """
    # This system prompt is optimized for fill-in-the-blank tasks.
    system_prompt = """You are an automated data extraction service. Your only function is to provide the precise Arabic terms that fill the blanks in the user's text. Your output must strictly be the answer(s) and nothing else. Follow these examples perfectly.

Example 1:
User: "املأ الفراغات: يتم تشخيص المرض بـ ANA و DNA."
Your perfect response: "الأجسام المضادة للنواة, الحمض النووي الريبي"

Example 2:
User: "املأ الفراغ: يعرف هذا بـ _____."
Your perfect response: "التهاب المفاصل"

Now, process the user's request based on these exact rules and examples. Provide only the text for the blank(s)."""

    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            # MODIFIED: API call now uses the deepseek model with streaming enabled.
            completion = client.chat.completions.create(
              model="deepseek-ai/deepseek-r1-0528",
              messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
              temperature=0.6,
              top_p=0.7,
              max_tokens=4096,
              stream=True  # Streaming is enabled
            )

            # MODIFIED: Accumulate the response from the stream
            raw_response_text = ""
            for chunk in completion:
                # The 'reasoning_content' is ignored as we only want the final answer.
                if chunk.choices[0].delta.content is not None:
                    raw_response_text += chunk.choices[0].delta.content

            # The existing cleaning logic is applied to the fully accumulated response
            text_without_think_block = re.sub(r'<think>.*?</think>', '', raw_response_text, flags=re.DOTALL)
            arabic_only_filter = r'[^\u0600-\u06FF\u0660-\u0669,\s]'
            final_cleaned_text = re.sub(arabic_only_filter, '', text_without_think_block)

            return final_cleaned_text.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   -> An error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"   -> API Error after multiple retries: {e}")
                return "API_ERROR" # Return a placeholder on failure
    return "FAILED_AFTER_RETRIES" # Return a placeholder if all retries fail

# --- Main Execution ---
def main():
    """
    Main function to read questions from a blind test set, generate predictions,
    and save them to a CSV file.
    """
    try:
        # Read the TSV file without a header.
        # The questions are in the first column (index 0).
        df = pd.read_csv(INPUT_TSV, sep='\t', header=None)
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_TSV}' was not found.")
        print("Please make sure you have uploaded your 'subtask2_questions.tsv' file.")
        return

    print(f"Starting prediction generation for {len(df)} questions using deepseek-ai/deepseek-r1-0528...")
    predictions = []
    total_questions = len(df)

    # Loop through the DataFrame, accessing questions from the first column.
    for index, row in df.iterrows():
        # The question is in the first column (index 0)
        question = row[0]
        if pd.isna(question) or not str(question).strip():
            print(f"Skipping empty line at row {index + 1}/{total_questions}.")
            continue

        print(f"Processing question {index + 1}/{total_questions}...")
        answer = generate_answer(str(question))
        predictions.append(answer)
        print(f"   -> Generated Answer: {answer}")

        # Add a short delay to be respectful of API rate limits
        if index < total_questions - 1:
            time.sleep(1)

    # Save the generated predictions to a CSV file without a header or index.
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(OUTPUT_CSV, header=False, index=False)
    print("\n" + "="*50)
    print(f"✅ Successfully generated all predictions.")
    print(f"📄 Results saved to '{OUTPUT_CSV}'.")
    print("="*50)

    # Offer to download the file automatically in Colab
    try:
        files.download(OUTPUT_CSV)
        print(f"\n🚀 Downloading '{OUTPUT_CSV}'...")
    except (NameError, ImportError):
        print(f"\nTo download '{OUTPUT_CSV}', please use the file browser on the left.")

if __name__ == "__main__":
    main()

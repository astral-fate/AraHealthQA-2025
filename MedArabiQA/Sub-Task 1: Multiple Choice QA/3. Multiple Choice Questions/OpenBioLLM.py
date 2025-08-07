import os
import re
import pandas as pd
import time
# Import the userdata module for Google Colab secrets
from google.colab import userdata
from sklearn.metrics import accuracy_score
# Import libraries for local Hugging Face model inference
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# --- MODIFIED: File paths and column names updated for your dataset ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/multiple-choice-questions.csv'
# Using a new output file name to reflect the new input data
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/mcq/predictions_OpenBioLLM_8B_mcq.csv'

# --- MODIFIED: Column names updated to match your CSV file ---
QUESTION_COLUMN = 'Question'
ANSWER_COLUMN = 'Answer'
# The 'Category' column from your file is available in the dataframe but not used in this script's logic.
# --- End of Modifications ---


# --- Chain of Thought & Few-Shot Prompting Configuration (Using pure Arabic format) ---
SYSTEM_PROMPT = """أنت خبير طبي ومساعد امتحانات دقيق. مهمتك هي حل سؤال متعدد الخيارات باللغة العربية.
أولاً، ستقوم بعملية تفكير خطوة بخطوة. قم بتحليل السؤال الطبي، وتقييم كل خيار (أ, ب, ج, د, هـ)، وشرح أسباب اختيارك للإجابة الصحيحة.
ثانياً، بعد شرح أسبابك، يجب عليك تقديم الإجابة النهائية في سطر جديد بالتنسيق التالي:
الإجابة النهائية: [الحرف العربي للخيار الصحيح]

هذه العملية المكونة من خطوتين إلزامية. يجب أن تكون إجابتك بأكملها باللغة العربية.
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
        "content": """**التفكير خطوة بخطوة:**
1.  **تحليل السؤال**: يسأل السؤال عن دلالة انخفاض أو غياب "الرجفان اللمسي" (tactile fremitus) في حالة "الانصباب الجنبي" (pleural effusion) وعن الآلية المسببة لذلك. الرجفان اللمسي هو الاهتزاز الذي يمكن الشعور به على جدار الصدر أثناء الكلام.
2.  **تقييم الخيارات**:
    * **أ. تراكم السوائل؛ عزل الصوت**: الانصباب الجنبي هو بالفعل تراكم للسوائل في الغشاء الجنبي. هذا السائل يعمل كعازل، مما يمنع انتقال اهتزازات الصوت من الرئة إلى جدار الصدر. هذا يتطابق تمامًا مع находة انخفاض الرجفان اللمسي.
    * **ب. احتباس الهواء؛ انهيار الحويصلات الهوائية**: هذا يصف حالة استرواح الصدر (pneumothorax) أو انخماص الرئة (atelectasis)، والتي لها موجودات فيزيائية مختلفة.
    * **ج. التليف؛ انخفاض مرونة الرئة**: التليف الرئوي (Pulmonary fibrosis) يزيد من كثافة أنسجة الرئة، مما قد يؤدي إلى زيادة الرجفان اللمسي، وليس انخفاضه.
    * **د. نمو الورم؛ انسداد الشعب الهوائية**: قد يسبب الورم انصبابًا جنبيًا، لكن السبب المباشر لانخفاض الرجفان في هذه الحالة هو السائل نفسه الذي يعزل الصوت. الخيار "أ" يصف الآلية الفيزيائية المباشرة بشكل أفضل.
3.  **الاستنتاج**: الخيار الأكثر دقة هو أن تراكم السوائل هو ما يسبب عزل الصوت، مما يؤدي إلى انخفاض الرجفان اللمسي.

الإجابة النهائية: أ"""
    }
]


# --- Function to Generate Answers using Local Hugging Face Model ---
def generate_answer(question, pipeline):
    """
    Generates an answer using a local transformers pipeline.
    """
    # Construct the message list for the chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    final_instruction = "الآن، اتبع التعليمات بدقة. ابدأ بالتفكير خطوة بخطوة ثم اختتم إجابتك بـ 'الإجابة النهائية: ' متبوعًا بالحرف الصحيح فقط."
    messages.append({"role": "user", "content": f"{question}\n\n{final_instruction}"})

    try:
        # Apply the chat template to create the full prompt
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Define terminators to stop generation cleanly
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate text using the pipeline
        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=False, # Use greedy decoding for more deterministic results
        )

        # Extract the generated text after the prompt
        response_text = outputs[0]["generated_text"][len(prompt):]

        # --- NEW & IMPROVED PARSING LOGIC ---
        
        # 1. Primary Method: Look for the English answer pattern the model is actually using.
        # This regex looks for "The answer is A" or "The correct answer is B", etc.
        english_match = re.search(r"The (?:correct )?answer is\s+([A-E])", response_text, re.IGNORECASE)
        if english_match:
            english_letter = english_match.group(1).upper()
            translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
            arabic_letter = translation_map.get(english_letter)
            if arabic_letter:
                # This is now the most likely successful path
                return arabic_letter
            
        # 2. Fallback: Look for the original Arabic format, just in case the model complies.
        match = re.search(r"الإجابة النهائية:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if match:
            return match.group(1)

        # 3. Fallback: Look for the original English "Final Answer" format.
        english_final_answer_match = re.search(r"Final Answer:\s*([A-E])", response_text, re.IGNORECASE)
        if english_final_answer_match:
            english_letter = english_final_answer_match.group(1).upper()
            translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
            arabic_letter = translation_map.get(english_letter)
            if arabic_letter:
                print(f"  -> Found English 'Final Answer:' format, translated '{english_letter}' to '{arabic_letter}'.")
                return arabic_letter

        # If all parsing fails, return an empty string
        print(f"  -> Warning: Could not deduce answer from response: '{response_text}'. Recording as empty.")
        return ""

    except Exception as e:
        print(f"  -> An error occurred during model inference: {e}")
        return "INFERENCE_ERROR"



# --- Function to Evaluate MCQ Accuracy ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """Calculates and prints the accuracy of the model's predictions, normalizing Alif."""
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

    # Helper function to normalize different forms of Alif
    def normalize_alif(letter):
        # Replaces hamza forms (أ, إ, آ) with the plain Alif (ا)
        return str(letter).replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    # Normalize both predictions and ground truths before comparison
    normalized_predictions = [normalize_alif(p) for p in valid_predictions]
    normalized_ground_truths = [normalize_alif(g) for g in valid_ground_truths]

    # Calculate accuracy using the normalized lists
    accuracy = accuracy_score(normalized_ground_truths, normalized_predictions)
    correct_predictions = int(accuracy * len(normalized_predictions))
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
    # --- Login to Hugging Face and initialize the local model pipeline ---
    try:
        # Login to Hugging Face Hub to download the model
        hf_token = userdata.get('HF_TOKEN')
        login(token=hf_token)
        print("✅ Successfully logged into Hugging Face Hub.")

        # Define the model ID
        model_id = "aaditya/Llama3-OpenBioLLM-8B"
        print(f"🚀 Initializing local model pipeline: {model_id}")

        # Explicitly set device to cuda if available, otherwise cpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {device}")

        # Load the tokenizer and set the chat template manually
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # This is the standard Llama 3 chat template
        llama3_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ bos_token + '<|start_header_id|>system<|end_header_id|>\n\n' + messages[0]['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        tokenizer.chat_template = llama3_template

        # Create the transformers pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer, # Pass the updated tokenizer
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        print("✅ Hugging Face pipeline initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize the Hugging Face pipeline. Error: {e}")
        print("Please ensure you have set 'HF_TOKEN' in Colab secrets and that the Colab runtime has a GPU.")
        return

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
        # Also include empty strings in the rerun list
        failed_indices = [i for i, p in enumerate(predictions) if p in error_codes_to_rerun or pd.isna(p) or p == ""]

        if not failed_indices:
            print("✅ No failed questions found to rerun. Proceeding directly to evaluation.")
        else:
            print(f"⚠️ Found {len(failed_indices)} failed questions. Starting rerun process...")
            for index in failed_indices:
                question = df.loc[index, QUESTION_COLUMN]
                print(f"Rerunning question {index + 1}/{len(df)}...")
                new_answer = ""
                # Add retry logic
                for attempt in range(2): # Try up to 2 times
                    print(f"  Attempt {attempt + 1}...")
                    new_answer = generate_answer(question, pipeline)
                    if new_answer and new_answer != "INFERENCE_ERROR":
                        break # Got a valid answer, no need to retry
                    if attempt == 0: # Only print retry message on the first failure
                        print(f"  -> Attempt 1 failed. Retrying...")
                predictions[index] = new_answer
                ground_truth_letter = str(df.loc[index, ANSWER_COLUMN]).strip()[0]
                print(f"  -> Ground Truth: {ground_truth_letter} | Final Predicted Letter: {new_answer}")

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
            answer_letter = ""
            # Add retry logic
            for attempt in range(2): # Try up to 2 times
                print(f"  Attempt {attempt + 1}...")
                answer_letter = generate_answer(question, pipeline)
                if answer_letter and answer_letter != "INFERENCE_ERROR":
                    break # Got a valid answer, no need to retry
                if attempt == 0: # Only print retry message on the first failure
                    print(f"  -> Attempt 1 failed. Retrying...")
            predictions.append(answer_letter)
            ground_truth_letter = str(row[ANSWER_COLUMN]).strip()[0] if str(row[ANSWER_COLUMN]).strip() else "N/A"
            print(f"  -> Ground Truth: {ground_truth_letter} | Final Predicted Letter: {answer_letter}")

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

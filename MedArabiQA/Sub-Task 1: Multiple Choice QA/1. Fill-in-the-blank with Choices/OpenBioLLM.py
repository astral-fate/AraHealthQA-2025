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

# --- File paths and column names (Updated for your data) ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/fill-in-the-blank-choices.csv'
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_fill-in-the-blank_OpenBioLLM_8B.csv'

# --- Column names (Set for your data) ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'
# --- NEW: Added a column name for our predictions to manage state ---
PREDICTED_COLUMN = 'Predicted_Answer'


# --- Chain of Thought & Few-Shot Prompting Configuration ---
# This system prompt is now in Arabic to guide the model more effectively.
SYSTEM_PROMPT = """أنت خبير طبي ومساعد امتحانات دقيق. مهمتك هي حل سؤال متعدد الخيارات باللغة العربية.
أولاً، ستقوم بعملية تفكير خطوة بخطوة. قم بتحليل السؤال الطبي، وتقييم كل خيار (أ, ب, ج, د, هـ)، وشرح أسباب اختيارك للإجابة الصحيحة.
ثانياً، بعد شرح أسبابك، يجب عليك تقديم الإجابة النهائية في سطر جديد بالتنسيق التالي:
Final Answer: [الحرف العربي للخيار الصحيح]

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

Final Answer: أ"""
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
    final_instruction = "الآن، اتبع التعليمات بدقة. ابدأ بالتفكير خطوة بخطوة ثم اختتم إجابتك بـ 'Final Answer: ' متبوعًا بالحرف الصحيح فقط."
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

        # --- Parsing Logic ---
        # 1. Primary method: Look for the Arabic format
        match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if match:
            return match.group(1)

        # 2. Heuristic method: Look for Arabic conclusive phrases
        conclusive_phrases = [
            r"الخيار الصحيح هو\s*([\u0621-\u064A])", r"الإجابة الصحيحة هي\s*([\u0621-\u064A])",
            r"الاستنتاج هو أن الخيار\s*([\u0621-\u064A])", r"الخيار\s*([\u0621-\u064A])\s*هو الصحيح",
        ]
        for phrase in conclusive_phrases:
            match = re.search(phrase, response_text)
            if match:
                print(f"  -> Found answer using Arabic conclusive phrase heuristic.")
                return match.group(1)

        # 3. Fallback method: Look for English letters and translate them
        english_match = re.search(r"The correct answer is\s+([A-E])", response_text, re.IGNORECASE)
        if not english_match:
            english_match = re.search(r"Final Answer:\s*([A-E])", response_text, re.IGNORECASE)

        if english_match:
            english_letter = english_match.group(1).upper()
            translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
            arabic_letter = translation_map.get(english_letter)
            if arabic_letter:
                print(f"  -> Found English letter '{english_letter}', translated to '{arabic_letter}'.")
                return arabic_letter

        # 4. Last resort: Look for any mentioned Arabic letter
        option_mentions = re.findall(r"الخيار\s*([\u0621-\u064A])", response_text)
        if option_mentions:
            last_option = option_mentions[-1]
            print(f"  -> Found answer using last-mentioned option heuristic: '{last_option}'")
            return last_option

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
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes and pd.notna(p)]
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
    # Load from output file if it exists to resume, otherwise start from input
    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Found existing predictions file: '{OUTPUT_CSV}'. Loading to resume.")
        try:
            df = pd.read_csv(OUTPUT_CSV, encoding='utf-8')
        except Exception as e:
            print(f"An error occurred while reading the existing output CSV: {e}")
            return
    else:
        print(f"No existing output file found. Loading from input: '{INPUT_CSV}'.")
        try:
            df = pd.read_csv(INPUT_CSV, encoding='utf-8')
        except FileNotFoundError:
            print(f"Error: The input file '{INPUT_CSV}' was not found. Please check the path.")
            return
        except Exception as e:
            print(f"An error occurred while reading the input CSV: {e}")
            return
        # Add the prediction column for the first run
        df[PREDICTED_COLUMN] = ""


    # --- Logic to Run Predictions (Full or Rerun) ---
    df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Ensure the prediction column exists and fill NaNs with empty strings for processing
    if PREDICTED_COLUMN not in df.columns:
        df[PREDICTED_COLUMN] = ""
    df[PREDICTED_COLUMN] = df[PREDICTED_COLUMN].fillna("")

    # Identify questions that need a prediction
    error_codes_to_rerun = ["INFERENCE_ERROR", ""]
    indices_to_run = df[df[PREDICTED_COLUMN].isin(error_codes_to_rerun)].index

    if indices_to_run.empty:
        print("✅ No questions found to run. All answers are present. Proceeding to evaluation.")
    else:
        print(f"Found {len(indices_to_run)} questions to process. Starting prediction process...")
        start_time = time.time()
        
        for i, index in enumerate(indices_to_run):
            question = df.loc[index, QUESTION_COLUMN]
            print(f"Processing question {i + 1}/{len(indices_to_run)} (Overall index: {index})...")
            
            predicted_answer = ""
            # Add retry logic
            for attempt in range(2): # Try up to 2 times
                print(f"  Attempt {attempt + 1}...")
                predicted_answer = generate_answer(question, pipeline)
                if predicted_answer and predicted_answer != "INFERENCE_ERROR":
                    break # Got a valid answer, no need to retry
                if attempt == 0: # Only print retry message on the first failure
                    print(f"  -> Attempt 1 failed. Retrying...")
            
            # Update the DataFrame with the new prediction
            df.loc[index, PREDICTED_COLUMN] = predicted_answer
            
            ground_truth_letter = str(df.loc[index, ANSWER_COLUMN]).strip()[0]
            print(f"  -> Ground Truth: {ground_truth_letter} | Final Predicted Letter: {predicted_answer}")
            
            # Save the updated DataFrame to the output file after each prediction
            # This makes the process resumable in case of interruption
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

        end_time = time.time()
        total_duration = end_time - start_time
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        print("\n" + "="*50)
        print(f"✅ Prediction generation complete.")
        print(f"⏱️  Total time taken: {minutes} minutes and {seconds} seconds.")
        print(f"Successfully saved all predictions to '{OUTPUT_CSV}'.")
        print("="*50)


    # --- Final Evaluation ---
    # Convert columns to lists for the evaluation function
    predictions = df[PREDICTED_COLUMN].tolist()
    ground_truths = [str(ans).strip()[0] if pd.notna(ans) and str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
    
    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

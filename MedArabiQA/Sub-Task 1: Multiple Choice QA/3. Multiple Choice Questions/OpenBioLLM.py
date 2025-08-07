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
def generate_answer(question, pipeline, use_sampling=False):
    """
    Generates an answer using a local transformers pipeline.
    
    Args:
        question (str): The MCQ question to answer.
        pipeline (transformers.Pipeline): The initialized text-generation pipeline.
        use_sampling (bool): If True, uses temperature-based sampling for more varied outputs.
                             If False, uses greedy decoding for deterministic output.
                             
    Returns:
        tuple: A tuple containing the parsed Arabic letter (or "") and the full response text.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    final_instruction = "الآن، اتبع التعليمات بدقة. ابدأ بالتفكير خطوة بخطوة ثم اختتم إجابتك بـ 'الإجابة النهائية: ' متبوعًا بالحرف الصحيح فقط."
    messages.append({"role": "user", "content": f"{question}\n\n{final_instruction}"})

    try:
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # --- MODIFIED: Use different generation settings for reruns ---
        generation_kwargs = {"do_sample": False} # Default: deterministic
        if use_sampling:
            # For reruns: use sampling to get a different response
            generation_kwargs = {"do_sample": True, "temperature": 0.6}

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            **generation_kwargs
        )

        response_text = outputs[0]["generated_text"][len(prompt):]
        parsed_letter = ""

        # --- MODIFIED: More robust parsing logic ---
        # 1. Primary Method: Look for the English answer pattern, allowing for optional quotes.
        english_match = re.search(r"The (?:correct )?answer is\s+'?([A-E])'?", response_text, re.IGNORECASE)
        if english_match:
            english_letter = english_match.group(1).upper()
            translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
            parsed_letter = translation_map.get(english_letter, "")
            return parsed_letter, response_text

        # 2. Fallback: Look for the original Arabic format.
        arabic_match = re.search(r"الإجابة النهائية:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if arabic_match:
            parsed_letter = arabic_match.group(1)
            return parsed_letter, response_text
        
        # 3. Last Resort Heuristic: Find the last mentioned option letter in the text
        option_mentions = re.findall(r"option\s+([A-E])", response_text, re.IGNORECASE)
        if option_mentions:
            english_letter = option_mentions[-1].upper()
            translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}
            parsed_letter = translation_map.get(english_letter, "")
            print(f"  -> Found answer using heuristic: last mentioned option '{english_letter}'.")
            return parsed_letter, response_text


        # If all parsing fails, return empty letter but the full response
        return "", response_text

    except Exception as e:
        error_message = f"An error occurred during model inference: {e}"
        print(f"  -> {error_message}")
        return "INFERENCE_ERROR", error_message



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
    try:
        hf_token = userdata.get('HF_TOKEN')
        login(token=hf_token)
        print("✅ Successfully logged into Hugging Face Hub.")

        model_id = "aaditya/Llama3-OpenBioLLM-8B"
        print(f"🚀 Initializing local model pipeline: {model_id}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        print("✅ Hugging Face pipeline initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize the Hugging Face pipeline. Error: {e}")
        return

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

    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Found existing prediction file: '{OUTPUT_CSV}'.")
        predictions_df = pd.read_csv(OUTPUT_CSV, header=None, encoding='utf-8', na_filter=False)
        predictions = predictions_df[0].astype(str).tolist()

        error_codes_to_rerun = ["INFERENCE_ERROR", ""]
        failed_indices = [i for i, p in enumerate(predictions) if p.strip() in error_codes_to_rerun]

        if not failed_indices:
            print("✅ No failed questions found to rerun. Proceeding directly to evaluation.")
        else:
            print(f"⚠️ Found {len(failed_indices)} failed questions. Starting rerun process...")
            for index in failed_indices:
                question = df.loc[index, QUESTION_COLUMN]
                print(f"Rerunning question {index + 1}/{len(df)}...")
                new_answer = ""
                
                # --- MODIFIED: Retry logic with sampling and full response printing ---
                for attempt in range(2): 
                    print(f"  Attempt {attempt + 1}...")
                    # Use sampling=True to get different reasoning
                    predicted_letter, full_response = generate_answer(question, pipeline, use_sampling=True)
                    
                    if predicted_letter and predicted_letter != "INFERENCE_ERROR":
                        new_answer = predicted_letter
                        break 
                    
                    # If parsing fails, print the full reasoning from the model
                    print("  -> Rerun failed to extract answer. LLM reasoning was:")
                    print("  " + "="*20 + " MODEL RESPONSE " + "="*20)
                    print(f"  {full_response.strip()}")
                    print("  " + "="*58)
                    if attempt == 0:
                        print(f"  -> Attempt 1 failed. Retrying with different sampling...")

                predictions[index] = new_answer
                ground_truth_letter = str(df.loc[index, ANSWER_COLUMN]).strip()[0]
                print(f"  -> Ground Truth: {ground_truth_letter} | Final Predicted Letter: {new_answer}")

            print("\n✅ Rerun complete. Saving updated results...")
            updated_predictions_df = pd.DataFrame(predictions)
            updated_predictions_df.to_csv(OUTPUT_CSV, header=False, index=False, encoding='utf-8')
            print(f"Successfully saved updated predictions to '{OUTPUT_CSV}'.")
    else:
        print(f"'{OUTPUT_CSV}' not found. Starting a full prediction run...")
        predictions = []
        total_questions = len(df)
        start_time = time.time()

        for index, row in df.iterrows():
            question = row[QUESTION_COLUMN]
            print(f"Processing question {index + 1}/{total_questions}...")
            # --- MODIFIED: Call generate_answer without sampling for the first run ---
            # We only need the letter here, so we discard the full response with `_`
            answer_letter, _ = generate_answer(question, pipeline, use_sampling=False)
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

    ground_truths = [str(ans).strip()[0] if str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

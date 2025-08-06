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
PREDICTED_COLUMN = 'Predicted_Answer'


# --- Chain of Thought & Few-Shot Prompting Configuration ---
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
            do_sample=False,
        )

        # Extract the generated text after the prompt
        response_text = outputs[0]["generated_text"][len(prompt):]

        # --- NEW: Robust Multi-Layer Parsing Logic ---
        translation_map = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه'}

        # Layer 1: Check for the ideal "Final Answer: [Letter]" format (Arabic or English)
        match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if match:
            print("  -> Found answer using 'Final Answer: [Arabic Letter]' format.")
            return match.group(1).strip()
        
        match = re.search(r"Final Answer:\s*([A-E])", response_text, re.IGNORECASE)
        if match:
            english_letter = match.group(1).upper()
            arabic_letter = translation_map.get(english_letter)
            print(f"  -> Found answer using 'Final Answer: [English Letter]' format ('{english_letter}'), translated to '{arabic_letter}'.")
            return arabic_letter

        # Layer 2: Flexible check for English phrases like "The answer is A)" or "The correct answer is 'B'."
        flexible_english_phrases = [
            r"(?:The correct answer is|The answer is|The final answer is)\s*['\"]?([A-E])['\"]?[\)\.]?",
            r"Option\s+([A-E])\s+is the correct answer"
        ]
        for phrase_regex in flexible_english_phrases:
            match = re.search(phrase_regex, response_text, re.IGNORECASE)
            if match:
                english_letter = match.group(1).upper()
                arabic_letter = translation_map.get(english_letter)
                print(f"  -> Found answer using flexible English phrase heuristic ('{english_letter}'), translated to '{arabic_letter}'.")
                return arabic_letter

        # Layer 3: Flexible check for hybrid phrases like "The correct answer is أ"
        match = re.search(r"(?:The correct answer is|The answer is)\s*['\"]?([\u0621-\u064A])['\"]?", response_text, re.IGNORECASE)
        if match:
            arabic_letter = match.group(1).strip()
            print(f"  -> Found answer using hybrid English phrase with Arabic letter heuristic: '{arabic_letter}'.")
            return arabic_letter

        # Layer 4: Check for standard Arabic conclusive phrases
        conclusive_phrases = [
            r"الخيار الصحيح هو\s*([\u0621-\u064A])", r"الإجابة الصحيحة هي\s*([\u0621-\u064A])",
            r"الاستنتاج هو أن الخيار\s*([\u0621-\u064A])", r"الخيار\s*([\u0621-\u064A])\s*هو الصحيح",
        ]
        for phrase in conclusive_phrases:
            match = re.search(phrase, response_text)
            if match:
                print(f"  -> Found answer using Arabic conclusive phrase heuristic.")
                return match.group(1).strip()

        # Layer 5: Last resort - Find the last standalone English option letter mentioned
        option_mentions = re.findall(r"\b([A-E])[\)\.]?\b", response_text)
        if option_mentions:
            last_english_letter = option_mentions[-1].upper()
            arabic_letter = translation_map.get(last_english_letter)
            print(f"  -> Found answer using last-mentioned English letter fallback ('{last_english_letter}'), translated to '{arabic_letter}'.")
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
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes and pd.notna(p)]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate. Check for widespread inference or parsing errors.")
        return

    def normalize_alif(letter):
        return str(letter).replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    normalized_predictions = [normalize_alif(p) for p in valid_predictions]
    normalized_ground_truths = [normalize_alif(g) for g in valid_ground_truths]

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

        # --- FIX: Manually set the Llama 3 chat template ---
        # This template is required for the apply_chat_template function to work correctly.
        llama3_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ bos_token + '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ bos_token + '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        if tokenizer.chat_template is None:
            print("  -> Tokenizer chat template not set. Applying Llama 3 template.")
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
        print("Please ensure you have set 'HF_TOKEN' in Colab secrets and that the Colab runtime has a GPU.")
        return

    # --- Load and Prepare Data ---
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
            df[PREDICTED_COLUMN] = ""
        except FileNotFoundError:
            print(f"Error: The input file '{INPUT_CSV}' was not found. Please check the path.")
            return
        except Exception as e:
            print(f"An error occurred while reading the input CSV: {e}")
            return

    # --- Logic to Run Predictions ---
    df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if PREDICTED_COLUMN not in df.columns:
        df[PREDICTED_COLUMN] = ""
    df[PREDICTED_COLUMN] = df[PREDICTED_COLUMN].fillna("")

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
            for attempt in range(2):
                print(f"  Attempt {attempt + 1}...")
                predicted_answer = generate_answer(question, pipeline)
                if predicted_answer and predicted_answer != "INFERENCE_ERROR":
                    break
                if attempt == 0:
                    print(f"  -> Attempt 1 failed. Retrying...")
            
            df.loc[index, PREDICTED_COLUMN] = predicted_answer
            
            ground_truth_letter = str(df.loc[index, ANSWER_COLUMN]).strip()[0]
            print(f"  -> Ground Truth: {ground_truth_letter} | Final Predicted Letter: {predicted_answer}")
            
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
    predictions = df[PREDICTED_COLUMN].tolist()
    ground_truths = [str(ans).strip()[0] if pd.notna(ans) and str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
    
    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

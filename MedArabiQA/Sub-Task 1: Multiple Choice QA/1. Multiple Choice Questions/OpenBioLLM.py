import os
import re
import pandas as pd
import time
from getpass import getpass
from sklearn.metrics import accuracy_score
import gc

# --- Import necessary libraries for local inference ---
try:
    from transformers import pipeline, BitsAndBytesConfig
    import torch
except ImportError:
    print("Please install necessary libraries: pip install transformers torch accelerate bitsandbytes")
    exit()


# --- File paths and column names ---
INPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/data/fill-in-the-blank-choices.csv'
# Changed output file name for the new model
OUTPUT_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/final_result/predictions_fitb_choices_OpenBioLLM.csv'

# --- Column names ---
QUESTION_COLUMN = 'Question - Arabic'
ANSWER_COLUMN = 'Answer - Arabic'


# --- Chain of Thought & Few-Shot Prompting Configuration ---
# Using the specific system prompt for OpenBioLLM
SYSTEM_PROMPT = """You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."""

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


# --- Function to Generate Answers ---
def generate_answer(question, pipe):
    """
    Generates an answer using the OpenBioLLM pipeline.
    Uses the apply_chat_template method and includes intelligent parsing.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    # The FEW_SHOT_EXAMPLES are not used here as we are now using a different prompting strategy
    # If the model struggles, we can add them back. For now, let's follow the user's snippet structure.
    messages.append({"role": "user", "content": question})

    try:
        # Apply the chat template to create the prompt string
        prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Define terminators for the model's generation
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generate the text
        outputs = pipe(
            prompt,
            max_new_tokens=1024, # Increased for detailed reasoning
            eos_token_id=terminators,
            do_sample=False, # Use False for deterministic, exam-like answers
        )
        
        # Extract only the newly generated text
        response_text = outputs[0]["generated_text"][len(prompt):].strip()
        
        # --- Intelligent Parsing Logic ---
        # Method 1: Check for the standard 'Final Answer:' format first.
        match = re.search(r"Final Answer:\s*([\u0621-\u064A])", response_text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Method 2: If standard format fails, parse the reasoning text.
        print(f"  -> 'Final Answer' format not found. Attempting to parse reasoning...")
        
        # Heuristic 2.1: Look for explicit conclusive phrases.
        conclusive_phrases = [
            r"الخيار الصحيح هو\s*([\u0621-\u064A])",
            r"الإجابة الصحيحة هي\s*([\u0621-\u064A])",
            r"الاستنتاج هو أن الخيار\s*([\u0621-\u064A])",
            r"الخيار\s*([\u0621-\u064A])\s*هو الصحيح",
        ]
        for phrase in conclusive_phrases:
            match = re.search(phrase, response_text)
            if match:
                print(f"  -> Found answer using conclusive phrase heuristic.")
                return match.group(1)

        # Heuristic 2.2: Assume the last mentioned option is the intended answer.
        option_mentions = re.findall(r"الخيار\s*([\u0621-\u064A])", response_text)
        if option_mentions:
            last_option = option_mentions[-1]
            print(f"  -> Found answer using last-mentioned option heuristic: '{last_option}'")
            return last_option
        
        print(f"  -> Warning: Could not deduce answer from response: '{response_text}'. Recording as empty.")
        return ""
            
    except Exception as e:
        print(f"  -> An error occurred during model inference: {e}")
        return "INFERENCE_ERROR"


# --- Function to Evaluate MCQ Accuracy (with Normalization) ---
def evaluate_mcq_accuracy(predictions, ground_truths):
    """
    Calculates and prints the accuracy of the MCQ predictions.
    Normalizes different forms of the Arabic letter Alif before comparison.
    """
    print("\n" + "="*50)
    print("🚀 Starting Evaluation...")
    print("="*50)
    
    error_codes = ["INFERENCE_ERROR", ""]
    valid_indices = [i for i, p in enumerate(predictions) if p not in error_codes]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]

    if not valid_predictions:
        print("No valid predictions to evaluate.")
        return

    # --- Normalization Logic ---
    def normalize_alif(letter):
        """Replaces all variants of Alif (أ, إ, آ) with a plain Alif (ا)."""
        return letter.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    # Apply normalization to both predictions and ground truths
    normalized_predictions = [normalize_alif(p) for p in valid_predictions]
    normalized_ground_truths = [normalize_alif(g) for g in valid_ground_truths]
    # --- End of Normalization Logic ---

    accuracy = accuracy_score(normalized_ground_truths, normalized_predictions)
    correct_predictions = sum(p == g for p, g in zip(normalized_ground_truths, normalized_predictions))
    
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


# --- Main Execution ---
def main():
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
        print(f"Output file '{OUTPUT_CSV}' already exists. Please remove it or rename it to run a new generation.")
        return
        
    # --- Model Loading ---
    print("="*50)
    print("🚀 Initializing local model: aaditya/OpenBioLLM-Llama3-8B")
    print("This may take a few minutes...")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. A GPU is required for this model.")
        return
    
    try:
        print("Clearing GPU cache before model loading...")
        gc.collect()
        torch.cuda.empty_cache()
        
        # Proactively use 4-bit quantization for memory safety
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model_id = "aaditya/OpenBioLLM-Llama3-8B"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config}
        )
    except Exception as e:
        print(f"❌ Failed to load the model. Error: {e}")
        return

    print("✅ Model loaded successfully.")
    
    # --- Prediction Generation ---
    predictions = []
    total_questions = len(df)
    start_time = time.time()

    for index, row in df.iterrows():
        question = row[QUESTION_COLUMN]
        print(f"Processing question {index + 1}/{total_questions}...")
        answer_letter = generate_answer(question, pipe)
        predictions.append(answer_letter)
        
        ground_truth_letter = str(row[ANSWER_COLUMN]).strip()[0] if str(row[ANSWER_COLUMN]).strip() else "N/A"
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

    # --- Final Evaluation ---
    ground_truths = [str(ans).strip()[0] if str(ans).strip() else "INVALID_TRUTH" for ans in df[ANSWER_COLUMN].tolist()]
    evaluate_mcq_accuracy(predictions, ground_truths)


if __name__ == "__main__":
    main()

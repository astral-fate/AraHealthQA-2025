import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import warnings
import re

# Suppress future warnings from seaborn for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Mount Google Drive ---
try:
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    print(f"❌ Error mounting Google Drive: {e}")
    exit()

# --- 1. Curated Data from All "MCQ with Bias" Logs ---
# This list contains the final, definitive results for each unique model experiment.
mcq_bias_model_data = [
    {
        'model_name': 'Qwen (CoT)',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_NVIDIA_qwen.csv',
        'accuracy': 67.00,
        'time_minutes': 81 + 41/60
    },
    {
        'model_name': 'DeepSeek 70B (Groq)',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_groq_deepseek.csv',
        'accuracy': 53.54, # Corrected from the final log
        'time_minutes': 15 + 49/60
    },
    {
        'model_name': 'MedGemma (Local)',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_medgemma_quantized.csv',
        'accuracy': 46.00,
        'time_minutes': None # Time not logged in this run
    },
    {
        'model_name': 'Colosseum',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_colosseum.csv',
        'accuracy': 45.00,
        'time_minutes': 13 + 36/60
    },
    {
        'model_name': 'Llama3 70B (CoT)',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_lamma.csv',
        'accuracy': 40.00,
        'time_minutes': 20 + 15/60 # Updated time
    },
    {
        'model_name': 'Palmyra-Med',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_NVIDIA_palmyra.csv',
        'accuracy': 33.33,
        'time_minutes': 10 + 46/60
    },
     {
        'model_name': 'BiMediX2 (vLLM)',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/BiMediX2_mcq_bias_answers_v2.csv',
        'accuracy': 31.00,
        'time_minutes': 0.53
    },
    {
        'model_name': 'BioMistral (Fallback)',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_biomistral_noshot.csv',
        'accuracy': 23.00,
        'time_minutes': 41 + 45/60 # Updated time
    },
    {
        'model_name': 'OpenBioLLM 8B (Local)',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_OpenBioLLM_8B_local.csv',
        'accuracy': 18.07, # Corrected from the final log
        'time_minutes': None # Time not logged in this run
    },
    {
        'model_name': 'DeepSeek 8B',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_deepseek.csv',
        'accuracy': 18.68,
        'time_minutes': 16 + 27/60
    },
    {
        'model_name': 'Mixtral',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_mixtral.csv',
        'accuracy': 18.00,
        'time_minutes': 14 + 28/60
    },
    {
        'model_name': 'Gemini',
        'csv_path': '/content/drive/MyDrive/AraHealthQA/mcqbias/gemini_mcq.csv',
        'accuracy': 73.00,
        'time_minutes': 4.0
    },

]

models_df = pd.DataFrame(mcq_bias_model_data)
# Add MedGemma time
models_df.loc[models_df['model_name'] == 'MedGemma (Local)', 'time_minutes'] = 180

# --- 2. Data Loading and Preprocessing ---
try:
    original_data_path = '/content/drive/MyDrive/AraHealthQA/t2t1/data/multiple-choice-withbias.csv'
    original_df = pd.read_csv(original_data_path)
    if 'Category' not in original_df.columns:
        print(f"❌ FATAL ERROR: Column 'Category' not found in {original_data_path}.")
        exit()
    print(f"✅ Original data loaded from '{original_data_path}'.")
except FileNotFoundError:
    print(f"❌ Error: The file '{original_data_path}' was not found.")
    exit()

def extract_answer_letter(text):
    if not isinstance(text, str): return None
    match = re.search(r'([\u0621-\u064A])', text)
    return match.group(1) if match else None

def normalize_alif(letter):
    if not isinstance(letter, str): return letter
    return letter.strip().replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

# Correctly parse the 'Answer' column
original_df['Answer_Letter'] = original_df['Answer'].apply(extract_answer_letter)
original_df['Answer_Normalized'] = original_df['Answer_Letter'].apply(normalize_alif)

# --- 3. Error Analysis ---
all_errors = []
for index, model in models_df.iterrows():
    try:
        preds_df = pd.read_csv(model['csv_path'], header=None, names=['Prediction'])
        combined_df = original_df.join(preds_df)
        combined_df['Prediction_Normalized'] = combined_df['Prediction'].apply(normalize_alif)
        combined_df['is_correct'] = combined_df['Answer_Normalized'] == combined_df['Prediction_Normalized']
        errors = combined_df[~combined_df['is_correct']].copy()
        errors['model_name'] = model['model_name']
        all_errors.append(errors)
        print(f"✅ Processed errors for {model['model_name']}.")
    except FileNotFoundError:
        print(f"⚠️ Warning: Prediction file not found for {model['model_name']}.")
    except Exception as e:
        print(f"⚠️ An error occurred while processing {model['model_name']}: {e}")

# --- 4. EDA and Visualizations ---
if all_errors:
    error_analysis_df = pd.concat(all_errors)
    error_counts = error_analysis_df.groupby(['model_name', 'Category']).size().reset_index(name='error_count')

    print("\n" + "="*50)
    print("📊 Final EDA for MCQ with Bias Task")
    print("="*50)

    total_errors_per_model = error_analysis_df.groupby('model_name').size().sort_values(ascending=True)
    print("\n--- Total Errors per Model ---")
    print(total_errors_per_model)

    total_errors_per_category = error_analysis_df.groupby('Category').size().sort_values(ascending=False)
    print("\n--- Top 5 Categories with Most Errors (All Models) ---")
    print(total_errors_per_category.head())

    # --- Generate Pivot Table for LaTeX ---
    top_models = models_df.nlargest(5, 'accuracy')['model_name'].tolist()
    top_categories = total_errors_per_category.head(5).index.tolist()

    pivot_df = error_analysis_df[
        error_analysis_df['model_name'].isin(top_models) &
        error_analysis_df['Category'].isin(top_categories)
    ]

    error_pivot_table = pivot_df.pivot_table(
        index='Category',
        columns='model_name',
        values='Question with Bias',
        aggfunc='count',
        fill_value=0
    ).loc[top_categories, top_models] # Ensure correct order

    print("\n--- Error Count Table for Top Models/Categories ---")
    print(error_pivot_table)
    print("\n--- LaTeX Formatted Table ---")
    print(error_pivot_table.to_latex())


    # --- Plot 1: Errors per Category with Improved Colors ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(18, 11))
    # Using a perceptually uniform and colorful palette like 'viridis'
    sns.barplot(data=error_counts, x='Category', y='error_count', hue='model_name', ax=ax1, palette='viridis')
    ax1.set_title('Model Errors per Category (MCQ with Bias Task)', fontsize=18, weight='bold')
    ax1.set_xlabel('Medical Category', fontsize=14)
    ax1.set_ylabel('Number of Incorrect Answers', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    ax1.legend(title='Model', fontsize=11)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Accuracy vs. Execution Time with Improved Colors ---
    plot_df = models_df.dropna(subset=['time_minutes'])

    fig2, ax2 = plt.subplots(figsize=(14, 9))
    # Using a bright and distinct palette like 'plasma'
    scatter = sns.scatterplot(data=plot_df, x='time_minutes', y='accuracy', hue='model_name', s=250, palette='plasma', ax=ax2)
    ax2.set_title('Model Accuracy vs. Execution Time (MCQ with Bias Task)', fontsize=18, weight='bold')
    ax2.set_xlabel('Execution Time (minutes)', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    for i, row in plot_df.iterrows():
        ax2.text(row['time_minutes'] + 0.5, row['accuracy'], row['model_name'], fontsize=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
else:
    print("\nCould not perform analysis. No valid error data was generated.")


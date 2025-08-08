# -*- coding: utf-8 -*-
"""
Final Consolidated EDA for MCQ with Bias Model Error Analysis

This script provides a comprehensive analysis of model performance on a biased
Multiple-Choice Question (MCQ) dataset. It dynamically calculates accuracy to
ensure data integrity and generates a set of key visualizations for a complete
overview.
"""

# --- 1. SETUP AND CONFIGURATION ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import warnings
import re

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# --- 2. MOUNT GOOGLE DRIVE ---
try:
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    print(f"❌ Error mounting Google Drive: {e}")
    exit()


# --- 3. MODEL AND DATA DEFINITIONS ---
# This list defines the models and their data paths. Accuracy is calculated later.
mcq_bias_model_data = [
    {'model_name': 'Qwen (CoT)', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_NVIDIA_qwen.csv', 'time_minutes': 81 + 41/60},
    {'model_name': 'DeepSeek 70B (Groq)', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_groq_deepseek.csv', 'time_minutes': 15 + 49/60},
    {'model_name': 'MedGemma (Local)', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_medgemma_quantized.csv', 'time_minutes': 180},
    {'model_name': 'Colosseum', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_colosseum.csv', 'time_minutes': 13 + 36/60},
    {'model_name': 'Llama3 70B (CoT)', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_lamma.csv', 'time_minutes': 20 + 15/60},
    {'model_name': 'Palmyra-Med', 'csv_path': '/content/drive/MyDrive/AraHealthQA/mcqbias/predictions_bias_test_NVIDIA_palmyra.csv', 'time_minutes': 10 + 46/60},
    {'model_name': 'BiMediX2 (vLLM)', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/BiMediX2_mcq_bias_answers_v2.csv', 'time_minutes': 0.53},
    {'model_name': 'BioMistral (Fallback)', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_biomistral_noshot.csv', 'time_minutes': 41 + 45/60},
    {'model_name': 'OpenBioLLM 8B (Local)', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_OpenBioLLM_8B_local.csv', 'time_minutes': None},
    {'model_name': 'Mixtral', 'csv_path': '/content/drive/MyDrive/AraHealthQA/t2t1/data/predictions_bias_test_mixtral.csv', 'time_minutes': 14 + 28/60},
    {'model_name': 'Gemini', 'csv_path': '/content/drive/MyDrive/AraHealthQA/mcqbias/gemini_bias.csv', 'time_minutes': 4.0},
]

models_df_initial = pd.DataFrame(mcq_bias_model_data)

# Load original ground truth data
try:
    original_data_path = '/content/drive/MyDrive/AraHealthQA/t2t1/data/multiple-choice-withbias.csv'
    original_df = pd.read_csv(original_data_path)
    total_questions = len(original_df)
    print(f"✅ Original data loaded from '{original_data_path}'. Total questions: {total_questions}")
except FileNotFoundError:
    print(f"❌ Error: The file '{original_data_path}' was not found.")
    exit()


# --- 4. PREPROCESSING & NORMALIZATION FUNCTIONS ---
def extract_answer_letter(text):
    """Extracts the first Arabic letter from a string."""
    if not isinstance(text, str): return None
    match = re.search(r'([\u0621-\u064A])', text)
    return match.group(1) if match else None

def normalize_alif(letter):
    """Normalizes different forms of the Arabic letter Alif."""
    if not isinstance(letter, str): return letter
    return letter.strip().replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

original_df['Answer_Letter'] = original_df['Answer'].apply(extract_answer_letter)
original_df['Answer_Normalized'] = original_df['Answer_Letter'].apply(normalize_alif)


# --- 5. DYNAMIC ACCURACY CALCULATION & ERROR ANALYSIS ---
all_errors = []
model_results = []

print("\n--- Starting Model Processing ---")
for index, model in models_df_initial.iterrows():
    try:
        preds_df = pd.read_csv(model['csv_path'], header=None, names=['Prediction'])
        if len(preds_df) != total_questions:
            print(f"⚠️ Warning: Prediction count mismatch for {model['model_name']}. Skipping.")
            continue

        combined_df = original_df.join(preds_df)
        combined_df['Prediction_Normalized'] = combined_df['Prediction'].apply(normalize_alif)
        combined_df['is_correct'] = combined_df['Answer_Normalized'] == combined_df['Prediction_Normalized']

        # Calculate accuracy and error count
        correct_answers = combined_df['is_correct'].sum()
        accuracy = (correct_answers / total_questions) * 100

        model_results.append({
            'model_name': model['model_name'],
            'accuracy': accuracy,
            'time_minutes': model['time_minutes'],
            'error_count': total_questions - correct_answers
        })

        # Collect errors for detailed analysis
        errors = combined_df[~combined_df['is_correct']].copy()
        errors['model_name'] = model['model_name']
        all_errors.append(errors)
        print(f"✅ Processed {model['model_name']} | Accuracy: {accuracy:.2f}%")

    except FileNotFoundError:
        print(f"⚠️ Warning: Prediction file not found for {model['model_name']}. Skipping.")
    except Exception as e:
        print(f"⚠️ An error occurred while processing {model['model_name']}: {e}")

# --- 6. FINAL DATA PREPARATION & REPORTING ---
if all_errors:
    # Create final DataFrames with calculated data
    models_df = pd.DataFrame(model_results)
    error_analysis_df = pd.concat(all_errors)
    error_counts_by_category = error_analysis_df.groupby(['model_name', 'Category']).size().reset_index(name='error_count_cat')

    print("\n" + "="*50)
    print("📊 Final EDA for MCQ with Bias Task (Calculated)")
    print("="*50)

    # --- Display Calculated Accuracies and Errors ---
    print("\n--- Model Performance Summary ---")
    print(models_df[['model_name', 'accuracy', 'error_count']].sort_values(by='accuracy', ascending=False).to_string(index=False))

    # --- Display Top Error Categories ---
    total_errors_per_category = error_analysis_df.groupby('Category').size().sort_values(ascending=False)
    print("\n--- Top 5 Categories with Most Errors (All Models) ---")
    print(total_errors_per_category.head())

    # --- Generate and Print LaTeX Table ---
    top_models = models_df.nlargest(5, 'accuracy')['model_name'].tolist()
    top_categories = total_errors_per_category.head(5).index.tolist()
    pivot_df = error_analysis_df[error_analysis_df['model_name'].isin(top_models) & error_analysis_df['Category'].isin(top_categories)]

    if not pivot_df.empty:
        error_pivot_table = pivot_df.pivot_table(index='Category', columns='model_name', values='Question with Bias', aggfunc='count', fill_value=0)
        error_pivot_table = error_pivot_table.reindex(index=top_categories, columns=top_models, fill_value=0) # Ensure order
        print("\n--- LaTeX Error Count Table for Top Models/Categories ---")
        print(error_pivot_table.to_latex())
    else:
        print("\n--- Could not generate pivot table (no overlapping data) ---")

# --- 7. VISUALIZATIONS ---
if all_errors:
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Heatmap of Error Distribution ---
    print("\nGenerating Heatmap of Error Distribution...")
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        error_pivot_table,
        annot=True,
        fmt="d",
        cmap='Reds',
        linewidths=.5,
        annot_kws={"size": 12}
    )
    plt.title('Error Distribution: Top 5 Models vs. Top 5 Categories', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Medical Category', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("error_heatmap.png")
    plt.show()

    # --- Plot 2: Bar Chart of Errors per Category ---
    print("\nGenerating Bar Chart of Errors per Category...")
    fig1, ax1 = plt.subplots(figsize=(18, 11))
    sns.barplot(data=error_counts_by_category, x='Category', y='error_count_cat', hue='model_name', ax=ax1, palette='viridis')
    ax1.set_title('Model Errors per Medical Category (MCQ with Bias Task)', fontsize=18, weight='bold')
    ax1.set_xlabel('Medical Category', fontsize=14)
    ax1.set_ylabel('Number of Incorrect Answers', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    ax1.legend(title='Model', fontsize=11)
    plt.tight_layout()
    plt.savefig("errors_per_category_barchart.png")
    plt.show()

    # --- Plot 3: Quadrant Analysis of Model Performance ---
    print("\nGenerating Quadrant Analysis of Model Performance...")
    plot_df = models_df.dropna(subset=['time_minutes', 'accuracy'])
    median_time = plot_df['time_minutes'].median()
    median_accuracy = plot_df['accuracy'].median()

    fig2, ax2 = plt.subplots(figsize=(16, 10))
    sns.scatterplot(data=plot_df, x='time_minutes', y='accuracy', hue='model_name', s=300, palette='plasma', ax=ax2)

    # Draw median lines for quadrant division
    ax2.axvline(median_time, color='grey', linestyle='--', lw=1.5)
    ax2.axhline(median_accuracy, color='grey', linestyle='--', lw=1.5)

    # Add quadrant labels
    ax2.text(ax2.get_xlim()[1], ax2.get_ylim()[1], ' High Accuracy, Slow ', fontsize=12, va='top', ha='right', backgroundcolor='w', color='darkorange')
    ax2.text(ax.get_xlim()[0], ax.get_ylim()[1], ' High Accuracy, Fast ', fontsize=12, va='top', ha='left', backgroundcolor='w', color='green')
    ax2.text(ax.get_xlim()[1], ax.get_ylim()[0], ' Low Accuracy, Slow ', fontsize=12, va='bottom', ha='right', backgroundcolor='w', color='red')
    ax2.text(ax.get_xlim()[0], ax.get_ylim()[0], ' Low Accuracy, Fast ', fontsize=12, va='bottom', ha='left', backgroundcolor='w', color='blue')

    # Set titles and labels
    ax2.set_title('Quadrant Analysis: Model Accuracy vs. Execution Time', fontsize=16, weight='bold', pad=20)
    ax2.set_xlabel('Execution Time (minutes)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(title='Model', bbox_to_anchor=(1.03, 1), loc='upper left')

    # Add labels to points
    for i, row in plot_df.iterrows():
        ax2.text(row['time_minutes'], row['accuracy'] + 0.8, row['model_name'], fontsize=9, ha='center')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("accuracy_vs_time_quadrant.png")
    plt.show()

else:
    print("\nCould not perform analysis. No valid error data was generated.")

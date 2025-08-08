import pandas as pd
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- File Paths ---
GROUND_TRUTH_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/test/subtask1_input_test.csv'
PREDICTIONS_CSV = '/content/drive/MyDrive/AraHealthQA/t2t1/test/llama3.3-70b.csv'

try:
    # Load the ground truth file (assuming it has a header)
    ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV)

    # Load the predictions file, specifying it has NO header
    predictions_df = pd.read_csv(PREDICTIONS_CSV, header=None)

    # --- Accuracy Calculation ---
    if len(ground_truth_df) == len(predictions_df):
        # Extract the correct letter from the 'Answer' column
        ground_truth_df['Correct_Letter'] = ground_truth_df['Answer'].str[0]

        # Since there's no header, access the first column of the predictions_df using its index: 0
        predicted_letters = predictions_df[0]

        # Create a new DataFrame for a clear comparison
        comparison_df = pd.DataFrame({
            'Correct_Letter': ground_truth_df['Correct_Letter'],
            'Predicted_Letter': predicted_letters,
            'Ground_Truth_Answer': ground_truth_df['Answer'],
        })

        # Determine if the prediction was correct
        comparison_df['is_correct'] = comparison_df['Correct_Letter'] == comparison_df['Predicted_Letter']

        # Calculate and print the accuracy
        accuracy = comparison_df['is_correct'].sum() / len(comparison_df)
        print(f"Accuracy: {accuracy:.2%}")

        # Display the first 10 rows for verification
        print("\n--- Comparison of the First 10 Answers ---")
        print(comparison_df.head(10))

    else:
        # This error message will now show the correct row counts
        print("Error: The number of rows in the files still do not match after the fix.")
        print(f"Ground truth file has {len(ground_truth_df)} rows.")
        print(f"Predictions file has {len(predictions_df)} rows.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nCould not find the files. Please make sure the paths are correct.")

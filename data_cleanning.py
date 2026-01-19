import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """
    Loads the dataset.
    """
    try:
        df = pd.read_csv(filepath, index_col=0)
        print(f"Successfully loaded data from {filepath}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

def validate_districts(df, txt_filepath, col_name='district'):
    """
    Validates that all districts in the DataFrame exist in the provided .txt file.
    Raises an error if unknown districts are found.
    """
    print("\n" + "="*30)
    print("DISTRICT VALIDATION")
    print("="*30)

    # 1. Check if the column exists in the CSV
    if col_name not in df.columns:
        print(f"Warning: Column '{col_name}' not found in dataset. Skipping validation.")
        # If your column might be named 'District' or 'DISTRICT', you might want to normalize names here
        return

    # 2. Load the reference list from txt file
    if not os.path.exists(txt_filepath):
        print(f"Error: Reference file '{txt_filepath}' not found.")
        return

    with open(txt_filepath, 'r') as f:
        # Create a set of valid districts (stripping newlines/spaces)
        valid_districts = set(line.strip() for line in f if line.strip())

    # 3. Get unique districts from the DataFrame
    # We convert to string and strip whitespace to ensure fair comparison
    df_districts = set(df[col_name].astype(str).str.strip().unique())

    # 4. Compare
    # Find values in DF that are NOT in the text file
    unknown_districts = df_districts - valid_districts

    if len(unknown_districts) > 0:
        print(f"CRITICAL ERROR: Found {len(unknown_districts)} districts in CSV not present in system file:")
        print(unknown_districts)
        raise ValueError("District validation failed. Dataset contains unknown districts.")
    else:
        print(f"Success: All {len(df_districts)} districts in the CSV match the system records.")

def inspect_nulls(df):
    """
    Strict null check. Raises error if nulls exist.
    """
    print("\n--- Null Value Validation ---")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls > 0:
        print(f"Found {total_nulls} missing values:")
        print(null_counts[null_counts > 0])
        raise ValueError("CRITICAL ERROR: Null values detected in the dataset.")
    else:
        print("Success: No null values found.")

def check_placeholders(df):
    """
    Checks for invalid placeholder values (e.g., '?', ' ').
    Raises a ValueError if any are found.
    """
    print("\n" + "="*30)
    print("PLACEHOLDER VALIDATION")
    print("="*30)

    placeholders = ['?', ' ']
    issues_found = False

    for ph in placeholders:
        # Check every column for this specific placeholder
        for col in df.columns:
            # We convert to string to ensure we catch symbols even in mixed columns
            count = (df[col].astype(str) == ph).sum()
            
            if count > 0:
                print(f"CRITICAL: Found {count} occurrences of '{ph}' in column '{col}'")
                issues_found = True

    if issues_found:
        print("\nFix required: Replace these placeholders (e.g., with NaN) before proceeding.")
        raise ValueError(f"Dataset contains invalid placeholders {placeholders}.")
    else:
        print("Success: No invalid placeholders ('?', ' ') found.")

def check_numeric_logic(df):
    """
    Checks for logical inconsistencies (negative values, zero percentages).
    """
    target_cols = ['year', 'month', 'Bio_bio_age_5_17', 'Bio_bio_age_17_']
    existing_cols = [c for c in target_cols if c in df.columns]
    
    if not existing_cols:
        return

    print("\n--- Numeric Logic Checks ---")
    
    # Check Negative Values
    try:
        neg_counts = (df[existing_cols] < 0).sum()
        # Only print columns with negative values
        if neg_counts.sum() > 0:
            print("Columns with negative values:")
            print(neg_counts[neg_counts > 0])
        else:
            print("No negative values found.")
    except TypeError:
        print("Warning: Columns contain non-numeric data (possibly '?'). Cannot check for negatives.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATA_FILE = r'Dataset_Cleaned.csv'
    DISTRICT_FILE = r'districts.txt' # The file containing valid district names
    DISTRICT_COL = 'district'        # The exact name of the column in your CSV
    # ---------------------

    df = load_data(DATA_FILE)
    
    if df is not None:
        inspect_nulls(df)
        check_placeholders(df)
        validate_districts(df, DISTRICT_FILE, col_name=DISTRICT_COL)
        check_numeric_logic(df)
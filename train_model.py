import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model(data_path, output_path='biometric_model_v1.pkl'):
    """
    Loads data, preprocesses features, trains a Random Forest model, 
    and saves the artifacts to a pickle file.
    """
    print(f"Loading data from {data_path}...")
    # Load your dataset
    # Ensure your CSV has columns: 'state', 'district', 'year', 'month', 'Bio_bio_age_5_17', 'Bio_bio_age_17_'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        return

    # --- Preprocessing ---
    print("Preprocessing data...")
    
    # Initialize encoders
    le_state = LabelEncoder()
    le_district = LabelEncoder()

    # Fit and transform categorical columns
    # We use .astype(str) to handle potential mixed types safely
    df['state_encoded'] = le_state.fit_transform(df['state'].astype(str))
    df['district_encoded'] = le_district.fit_transform(df['district'].astype(str))

    # Define Features and Targets
    features_list = ['state_encoded', 'district_encoded', 'year', 'month']
    X = df[features_list]
    y = df[['Bio_bio_age_5_17', 'Bio_bio_age_17_']]

    # --- Model Training ---
    print("Training Random Forest model on all data...")
    
    # Using the configuration from your second block (with OOB score and n_jobs=-1)
    model = RandomForestRegressor(
        n_estimators=100, 
        oob_score=True, 
        n_jobs=-1, 
        random_state=42
    )
    
    model.fit(X, y)

    # --- Evaluation ---
    oob_r2 = model.oob_score_
    print(f"Final Model Trained.")
    print(f"OOB Score (Estimated R²): {oob_r2:.4f}")

    # --- Model Export ---
    print(f"Saving artifacts to {output_path}...")
    
    artifacts = {
        'model': model,
        'le_state': le_state,
        'le_district': le_district,
        'features': features_list
    }

    joblib.dump(artifacts, output_path)
    print(f"✅ Saved successfully.")

if __name__ == "__main__":
    # REPLACE 'your_dataset.csv' with the actual path to your clean data file
    DATA_FILE = 'Dataset_Cleaned.csv' 
    
    train_and_save_model(DATA_FILE)
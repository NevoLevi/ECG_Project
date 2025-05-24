import pandas as pd
import numpy as np

def shuffle_patient_ids():
    """
    Shuffle only the patient_id column while keeping other data in place
    """
    print("Loading ECG hospital data...")
    df = pd.read_csv('data/ecg_hospital_data_updated.csv')
    
    print(f"Original data shape: {df.shape}")
    print(f"Original patient IDs (first 5): {df['patient_id'].head().tolist()}")
    
    # Get a copy of the patient_id column
    patient_ids = df['patient_id'].copy()
    
    # Shuffle the patient IDs
    shuffled_patient_ids = patient_ids.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Replace the patient_id column with shuffled values
    df['patient_id'] = shuffled_patient_ids
    
    print(f"Shuffled patient IDs (first 5): {df['patient_id'].head().tolist()}")
    
    # Save the updated file
    df.to_csv('data/ecg_hospital_data_updated.csv', index=False)
    print("✅ Patient IDs shuffled and saved successfully!")
    
    return df

if __name__ == "__main__":
    shuffle_patient_ids() 
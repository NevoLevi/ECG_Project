import pandas as pd
import numpy as np

def fix_and_transpose_csv():
    # Read the original CSV
    print("Reading original CSV...")
    df = pd.read_csv('frontend/public/cat_net_classification_report_20250609_042242.csv', index_col=0)
    
    print(f"Original shape: {df.shape}")
    print(f"Original index (first 10): {list(df.index[:10])}")
    
    # Extract heartbeat numbers and sort them numerically
    heartbeat_numbers = []
    for idx in df.index:
        # Extract number from heartbeat_X format
        heartbeat_num = int(idx.split('_')[1])
        heartbeat_numbers.append(heartbeat_num)
    
    # Create a mapping from heartbeat number to original index
    heartbeat_mapping = list(zip(heartbeat_numbers, df.index))
    # Sort by heartbeat number
    heartbeat_mapping_sorted = sorted(heartbeat_mapping, key=lambda x: x[0])
    
    print(f"Heartbeat range: {min(heartbeat_numbers)} to {max(heartbeat_numbers)}")
    print(f"Total heartbeats: {len(heartbeat_numbers)}")
    
    # Reorder the dataframe rows according to correct heartbeat order
    sorted_indices = [mapping[1] for mapping in heartbeat_mapping_sorted]
    df_sorted = df.loc[sorted_indices]
    
    print("Reordered heartbeats (first 10):", list(df_sorted.index[:10]))
    
    # Now transpose the dataframe
    # After transpose: rows = patients, columns = heartbeats
    df_transposed = df_sorted.transpose()
    
    print(f"Transposed shape: {df_transposed.shape}")
    print(f"Patients (rows): {df_transposed.shape[0]}")
    print(f"Heartbeats (columns): {df_transposed.shape[1]}")
    
    # Rename the index to be patient IDs (1-47 instead of 0-46)
    df_transposed.index = range(1, len(df_transposed) + 1)
    df_transposed.index.name = 'patient_id'
    
    # Rename columns to be heartbeat numbers (1, 2, 3... instead of heartbeat_1, heartbeat_2...)
    new_column_names = [heartbeat_mapping_sorted[i][0] for i in range(len(heartbeat_mapping_sorted))]
    df_transposed.columns = new_column_names
    
    print("Sample of transposed data:")
    print(df_transposed.iloc[:5, :10])  # First 5 patients, first 10 heartbeats
    
    # Save the fixed CSV
    output_file = 'frontend/public/cat_net_classification_report_fixed.csv'
    df_transposed.to_csv(output_file)
    print(f"\nFixed CSV saved to: {output_file}")
    
    # Verify a few patients to make sure the data looks right
    print("\nVerification - Patient 1 classifications (first 20 windows):")
    patient_1_data = df_transposed.loc[1].values[:20]
    print(list(patient_1_data))
    
    print("\nVerification - Patient 10 classifications (first 20 windows):")
    patient_10_data = df_transposed.loc[10].values[:20]
    print(list(patient_10_data))
    
    return df_transposed

if __name__ == "__main__":
    fixed_df = fix_and_transpose_csv() 
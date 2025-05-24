import pandas as pd
import numpy as np
import random

def create_ecg_hospital_data():
    print("Loading mitbih_train.csv...")
    df = pd.read_csv('data/mitbih_train.csv', header=None)
    
    # Separate ECG values (first 187 columns) and classifications (188th column)
    ecg_data = df.iloc[:, :187].values
    classifications = df.iloc[:, 187].values
    
    # Separate data by classification
    normal_indices = np.where(classifications == 0.0)[0]
    abnormal_indices = {
        1: np.where(classifications == 1.0)[0],
        2: np.where(classifications == 2.0)[0],
        3: np.where(classifications == 3.0)[0],
        4: np.where(classifications == 4.0)[0]
    }
    
    print(f"Normal sequences available: {len(normal_indices)}")
    print(f"Abnormal sequences available: {sum(len(indices) for indices in abnormal_indices.values())}")
    
    # Parameters
    total_patients = 300
    normal_only_patients = 290
    mixed_patients = total_patients - normal_only_patients
    sequences_per_patient = 100
    
    # Result list to store all patient data
    all_patients = []
    patient_info = []
    
    # Generate 290 patients with only normal sequences
    print(f"Generating {normal_only_patients} patients with only normal sequences...")
    for patient_id in range(1, normal_only_patients + 1):
        # Randomly select 100 normal sequences
        selected_indices = np.random.choice(normal_indices, sequences_per_patient, replace=False)
        
        # Concatenate the sequences
        patient_ecg = []
        for idx in selected_indices:
            patient_ecg.extend(ecg_data[idx])
        
        all_patients.append(patient_ecg)
        patient_info.append({
            'patient_id': patient_id,
            'type': 'normal_only',
            'normal_sequences': sequences_per_patient,
            'abnormal_sequences': 0,
            'abnormal_positions': []
        })
        
        if patient_id % 50 == 0:
            print(f"  Completed {patient_id}/{normal_only_patients} normal patients")
    
    # Generate 10 patients with mostly normal but some abnormal sequences
    print(f"Generating {mixed_patients} patients with mixed sequences...")
    for patient_id in range(normal_only_patients + 1, total_patients + 1):
        # Decide how many abnormal sequences to include (1-3 abnormal sequences)
        num_abnormal = random.randint(1, 3)
        num_normal = sequences_per_patient - num_abnormal
        
        # Choose positions for abnormal sequences (within first 50 positions)
        abnormal_positions = random.sample(range(50), num_abnormal)
        abnormal_positions.sort()
        
        # Select normal sequences
        selected_normal = np.random.choice(normal_indices, num_normal, replace=False)
        
        # Select abnormal sequences (randomly from classes 1, 2, 3)
        abnormal_classes = [1, 2, 3]  # Exclude class 4 as specified
        selected_abnormal = []
        abnormal_classes_used = []
        
        for _ in range(num_abnormal):
            # Randomly choose an abnormal class
            abnormal_class = random.choice(abnormal_classes)
            if len(abnormal_indices[abnormal_class]) > 0:
                idx = np.random.choice(abnormal_indices[abnormal_class])
                selected_abnormal.append(idx)
                abnormal_classes_used.append(abnormal_class)
            else:
                # Fallback to any available abnormal class
                for cls in abnormal_classes:
                    if len(abnormal_indices[cls]) > 0:
                        idx = np.random.choice(abnormal_indices[cls])
                        selected_abnormal.append(idx)
                        abnormal_classes_used.append(cls)
                        break
        
        # Create the sequence order
        sequence_order = ['normal'] * sequences_per_patient
        for pos in abnormal_positions:
            sequence_order[pos] = 'abnormal'
        
        # Build the patient ECG data
        patient_ecg = []
        normal_idx = 0
        abnormal_idx = 0
        
        for seq_type in sequence_order:
            if seq_type == 'normal':
                patient_ecg.extend(ecg_data[selected_normal[normal_idx]])
                normal_idx += 1
            else:
                patient_ecg.extend(ecg_data[selected_abnormal[abnormal_idx]])
                abnormal_idx += 1
        
        all_patients.append(patient_ecg)
        patient_info.append({
            'patient_id': patient_id,
            'type': 'mixed',
            'normal_sequences': num_normal,
            'abnormal_sequences': num_abnormal,
            'abnormal_positions': abnormal_positions,
            'abnormal_classes': abnormal_classes_used
        })
        
        if (patient_id - normal_only_patients) % 20 == 0:
            print(f"  Completed {patient_id - normal_only_patients}/{mixed_patients} mixed patients")
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    df_patients = pd.DataFrame(all_patients)
    
    # Add patient IDs as the first column
    patient_ids = [f"Patient_{i+1:03d}" for i in range(total_patients)]
    df_patients.insert(0, 'patient_id', patient_ids)
    
    # Save to CSV
    output_filename = 'data/ecg_hospital_data_updated.csv'
    print(f"Saving to {output_filename}...")
    df_patients.to_csv(output_filename, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    print(f"Total patients created: {total_patients}")
    print(f"Patients with only normal sequences: {normal_only_patients}")
    print(f"Patients with mixed sequences: {mixed_patients}")
    print(f"Sequences per patient: {sequences_per_patient}")
    print(f"ECG values per patient: {sequences_per_patient * 187} (18,700)")
    print(f"Estimated duration per patient: {(sequences_per_patient * 187) / 125:.1f} seconds")
    print(f"Output file size: {df_patients.shape}")
    
    # Detailed breakdown of mixed patients
    print(f"\nMixed patients breakdown:")
    abnormal_counts = {}
    for info in patient_info:
        if info['type'] == 'mixed':
            num_abnormal = info['abnormal_sequences']
            abnormal_counts[num_abnormal] = abnormal_counts.get(num_abnormal, 0) + 1
    
    for num_abnormal, count in sorted(abnormal_counts.items()):
        print(f"  {count} patients with {num_abnormal} abnormal sequence(s)")
    
    print(f"\nFile saved successfully: {output_filename}")
    return df_patients, patient_info

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    df_result, patient_details = create_ecg_hospital_data() 
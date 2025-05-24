import pandas as pd
import numpy as np

# Load MIT-BIH data
print("Loading MIT-BIH training data...")
df = pd.read_csv('data/mitbih_train.csv', header=None)

# Separate data by class
data_by_class = {}
for class_label in range(5):
    class_data = df[df.iloc[:, -1] == class_label].iloc[:, :-1].values
    data_by_class[class_label] = class_data
    print(f"Class {class_label}: {len(class_data)} sequences")

print(f"\nGenerating enhanced hospital data for 300 patients...")
print(f"Each patient will have 1000 sequences (187,000 samples)")
print(f"Estimated duration per patient: 1,496 seconds")

# Set random seed
np.random.seed(42)

patients = []
normal_data = data_by_class[0]
abnormal_classes = [1, 2, 3, 4]

for patient_idx in range(300):
    patient_id = f"Patient_{patient_idx+1:03d}"
    
    if patient_idx < 280:  # Normal patients
        print(f"Creating {patient_id} - All normal sequences")
        selected_indices = np.random.choice(len(normal_data), 1000, replace=True)
        patient_sequences = normal_data[selected_indices]
        
    else:  # Mixed patients  
        print(f"Creating {patient_id} - Mixed sequences with abnormalities")
        patient_sequences = []
        
        # 5-15% abnormal sequences
        abnormal_count = np.random.randint(50, 150)
        
        # 80% of abnormalities in first 300 sequences
        early_abnormal_count = int(abnormal_count * 0.8)
        late_abnormal_count = abnormal_count - early_abnormal_count
        
        early_positions = np.random.choice(300, early_abnormal_count, replace=False)
        late_positions = np.random.choice(range(300, 1000), late_abnormal_count, replace=False)
        abnormal_positions = set(list(early_positions) + list(late_positions))
        
        print(f"  - {abnormal_count} abnormal sequences")
        
        for seq_idx in range(1000):
            if seq_idx in abnormal_positions:
                # Add abnormal sequence
                abnormal_class = np.random.choice(abnormal_classes)
                abnormal_data = data_by_class[abnormal_class]
                sequence = abnormal_data[np.random.randint(len(abnormal_data))]
            else:
                # Add normal sequence
                sequence = normal_data[np.random.randint(len(normal_data))]
            
            patient_sequences.append(sequence)
        
        patient_sequences = np.array(patient_sequences)
    
    # Flatten sequences
    flattened_sequence = patient_sequences.flatten()
    patient_record = [patient_id] + flattened_sequence.tolist()
    patients.append(patient_record)
    
    if (patient_idx + 1) % 50 == 0:
        print(f"Progress: {patient_idx + 1}/300 patients created")

# Save to CSV
print(f"\nSaving enhanced hospital data...")
columns = ['patient_id'] + [f'ecg_{i}' for i in range(187000)]
df = pd.DataFrame(patients, columns=columns)
df.to_csv('data/ecg_hospital_data_enhanced.csv', index=False)

print(f"✅ Enhanced hospital data saved!")
print(f"Dataset shape: {df.shape}")
print(f"Total patients: {len(df)}")
print(f"Samples per patient: 187,000")
print(f"Duration per patient: 1,496 seconds") 
import pandas as pd
import numpy as np
import random
from datetime import datetime

def load_mitbih_data():
    """Load the MIT-BIH train dataset"""
    print("Loading MIT-BIH training data...")
    df = pd.read_csv('data/mitbih_train.csv', header=None)
    
    # Separate data by class (last column is the class)
    data_by_class = {}
    for class_label in range(5):  # Classes 0-4
        class_data = df[df.iloc[:, -1] == class_label].iloc[:, :-1].values
        data_by_class[class_label] = class_data
        print(f"Class {class_label}: {len(class_data)} sequences")
    
    return data_by_class

def create_enhanced_patient_data(data_by_class, num_patients=300, sequences_per_patient=1000):
    """
    Create enhanced patient data with:
    - 280 patients with all normal sequences
    - 20 patients with mixed sequences (abnormal mostly in first 300)
    """
    print(f"\nGenerating enhanced hospital data for {num_patients} patients...")
    print(f"Each patient will have {sequences_per_patient} sequences ({sequences_per_patient * 187} samples)")
    print(f"Estimated duration per patient: {(sequences_per_patient * 187) / 125:.1f} seconds")
    
    patients = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Classes: 0=Normal, 1=Supraventricular, 2=Ventricular, 3=Fusion, 4=Unknown
    normal_data = data_by_class[0]
    abnormal_classes = [1, 2, 3, 4]
    
    for patient_idx in range(num_patients):
        patient_id = f"Patient_{patient_idx+1:03d}"
        
        if patient_idx < 280:  # First 280 patients: ALL NORMAL
            print(f"Creating {patient_id} - All normal sequences")
            
            # Select 1000 random normal sequences
            selected_indices = np.random.choice(len(normal_data), sequences_per_patient, replace=True)
            patient_sequences = normal_data[selected_indices]
            
        else:  # Last 20 patients: MIXED with abnormalities
            print(f"Creating {patient_id} - Mixed sequences with abnormalities")
            
            patient_sequences = []
            
            # Determine how many abnormal sequences to include (5-15% of sequences)
            abnormal_count = np.random.randint(50, 150)  # 5-15% of 1000
            normal_count = sequences_per_patient - abnormal_count
            
            # Determine positions for abnormal sequences
            # Most abnormalities should appear in first 300 sequences
            early_abnormal_count = int(abnormal_count * 0.8)  # 80% in first 300
            late_abnormal_count = abnormal_count - early_abnormal_count
            
            # Positions for abnormal sequences
            early_positions = np.random.choice(300, early_abnormal_count, replace=False)
            late_positions = np.random.choice(range(300, sequences_per_patient), late_abnormal_count, replace=False)
            abnormal_positions = set(list(early_positions) + list(late_positions))
            
            print(f"  - {abnormal_count} abnormal sequences ({early_abnormal_count} in first 300, {late_abnormal_count} later)")
            print(f"  - {normal_count} normal sequences")
            
            # Generate sequences
            abnormal_iter = iter(abnormal_positions)
            next_abnormal_pos = next(abnormal_iter, None)
            
            for seq_idx in range(sequences_per_patient):
                if seq_idx == next_abnormal_pos:
                    # Add abnormal sequence
                    abnormal_class = np.random.choice(abnormal_classes)
                    abnormal_data = data_by_class[abnormal_class]
                    sequence = abnormal_data[np.random.randint(len(abnormal_data))]
                    next_abnormal_pos = next(abnormal_iter, None)
                else:
                    # Add normal sequence
                    sequence = normal_data[np.random.randint(len(normal_data))]
                
                patient_sequences.append(sequence)
            
            patient_sequences = np.array(patient_sequences)
        
        # Flatten all sequences into one long sequence
        flattened_sequence = patient_sequences.flatten()
        
        # Create patient record
        patient_record = [patient_id] + flattened_sequence.tolist()
        patients.append(patient_record)
        
        if (patient_idx + 1) % 50 == 0:
            print(f"Progress: {patient_idx + 1}/{num_patients} patients created")
    
    return patients

def save_enhanced_hospital_data(patients, sequences_per_patient=1000):
    """Save the enhanced patient data to CSV"""
    print(f"\nSaving enhanced hospital data...")
    
    # Create column names
    columns = ['patient_id'] + [f'ecg_{i}' for i in range(sequences_per_patient * 187)]
    
    # Create DataFrame
    df = pd.DataFrame(patients, columns=columns)
    
    # Save to CSV
    output_file = 'data/ecg_hospital_data_enhanced.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Enhanced hospital data saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Total patients: {len(df)}")
    print(f"Samples per patient: {sequences_per_patient * 187}")
    print(f"Duration per patient: {(sequences_per_patient * 187) / 125:.1f} seconds")
    print(f"Total file size: ~{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Show first few patient IDs
    print(f"Patient IDs: {df['patient_id'].head(10).tolist()}...")
    
    return output_file

def create_data_summary(output_file):
    """Create a summary of the generated data"""
    summary = f"""
# Enhanced ECG Hospital Data Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Specifications
- **Total Patients**: 300
- **Normal Patients**: 280 (patients 001-280)
- **Mixed Patients**: 20 (patients 281-300)
- **Sequences per Patient**: 1,000
- **Samples per Patient**: 187,000 (1,000 × 187)
- **Duration per Patient**: ~1,496 seconds (≈25 minutes)
- **Sampling Rate**: 125 Hz

## Patient Distribution
### Normal Patients (280)
- All 1,000 sequences are normal (class 0)
- Pure baseline cardiac rhythm

### Mixed Patients (20)
- 5-15% abnormal sequences mixed with normal sequences
- 80% of abnormalities appear in first 300 sequences
- 20% of abnormalities appear in remaining 700 sequences
- Abnormal classes include:
  - Class 1: Supraventricular Ectopic Beat
  - Class 2: Ventricular Ectopic Beat  
  - Class 3: Fusion Beat
  - Class 4: Unknown Beat

## File Details
- **Filename**: `{output_file}`
- **Format**: CSV with header row
- **Columns**: patient_id, ecg_0, ecg_1, ..., ecg_186999
- **Total Columns**: 187,001 (1 ID + 187,000 data points)

## Usage
This enhanced dataset is designed for:
- Real-time ECG monitoring simulation
- Long-term cardiac rhythm analysis
- Abnormality detection in extended monitoring periods
- Training models on realistic patient monitoring scenarios
"""
    
    summary_file = 'data/enhanced_data_summary.md'
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"✅ Data summary saved to: {summary_file}")
    return summary_file

def main():
    """Main function to generate enhanced ECG hospital data"""
    print("=" * 60)
    print("ENHANCED ECG HOSPITAL DATA GENERATOR")
    print("=" * 60)
    
    try:
        # Load MIT-BIH data
        data_by_class = load_mitbih_data()
        
        # Create enhanced patient data
        patients = create_enhanced_patient_data(
            data_by_class, 
            num_patients=300, 
            sequences_per_patient=1000
        )
        
        # Save to CSV
        output_file = save_enhanced_hospital_data(patients, sequences_per_patient=1000)
        
        # Create summary
        create_data_summary(output_file)
        
        print("\n" + "=" * 60)
        print("✅ ENHANCED ECG HOSPITAL DATA GENERATION COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error generating enhanced hospital data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
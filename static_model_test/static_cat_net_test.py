#!/usr/bin/env python3
"""
Static CAT-Net Model Testing Script
===================================

This script loads the actual CAT-Net.h5 model and tests it on the 47_Patients_Hospital.csv data.
For each patient, it runs classification on every window of 187 values and generates a report table.

Author: ECG Project Team
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom ChannelAttention layer for CAT-Net (exact replica from your model)
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8):
        super().__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.shared_mlp = tf.keras.models.Sequential([
            tf.keras.layers.Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=True),
            tf.keras.layers.Dense(filters, kernel_initializer="he_normal", use_bias=True)
        ])
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs):
        # inputs: (batch, time, channels)
        avg = self.avg_pool(inputs)
        max = self.max_pool(inputs)
        # shared MLP expects 2D
        avg = self.shared_mlp(avg)
        max = self.shared_mlp(max)
        attn = self.sigmoid(avg + max)[:, None, :]    # (batch, 1, channels)
        return inputs * attn

# Transformer encoder block as a custom layer for CAT-Net
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)
    
    def call(self, inputs):
        # Multi-Head Self-Attention
        attn_output = self.attention(inputs, inputs)
        x = self.layernorm1(inputs + attn_output)
        # Feed-Forward
        ffn_output = self.dense1(x)
        ffn_output = self.dense2(ffn_output)
        return self.layernorm2(x + ffn_output)
    
    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim
        })
        return config

# Keep the function version too for fallback
def transformer_encoder_block(x, embed_dim, num_heads, ff_dim):
    """Transformer encoder block with multi-head attention and feed-forward network"""
    # Multi-Head Self-Attention
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    # Feed-Forward
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn_output = tf.keras.layers.Dense(embed_dim)(ffn_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

class StaticCATNetTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.class_names = {
            0: 'Normal',
            1: 'Supraventricular', 
            2: 'Ventricular',
            3: 'Fusion',
            4: 'Unclassified'
        }
        self.class_names_short = {
            0: 'N',
            1: 'S', 
            2: 'V',
            3: 'F',
            4: 'Q'
        }
        
    def create_catnet(self, input_shape=(187,1), num_classes=5):
        """Recreate the exact CAT-Net architecture"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv1D(16, 21, padding="same", activation="relu")(inputs)
        x = ChannelAttention(16)(x)
        x = tf.keras.layers.MaxPool1D(3, strides=2, padding="same")(x)

        x = tf.keras.layers.Conv1D(32, 23, padding="same", activation="relu")(x)
        x = ChannelAttention(32)(x)
        x = tf.keras.layers.MaxPool1D(3, strides=2, padding="same")(x)

        x = tf.keras.layers.Conv1D(64, 25, padding="same", activation="relu")(x)
        x = ChannelAttention(64)(x)
        x = tf.keras.layers.AveragePooling1D(3, strides=2, padding="same")(x)

        x = tf.keras.layers.Conv1D(128, 27, padding="same", activation="relu")(x)
        x = ChannelAttention(128)(x)

        # Transformer encoder
        x = transformer_encoder_block(x, embed_dim=128, num_heads=4, ff_dim=256)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs, name="CAT_Net")
        return model

    def load_model_and_scaler(self):
        """Load the actual CAT-Net model and scaler parameters"""
        print("🔄 Loading CAT-Net Model and Scaler...")
        print("=" * 50)
        
        # Load CAT-Net model
        model_path = os.path.join('..', 'models', 'CAT-Net.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CAT-Net model not found at {model_path}")
            
        try:
            print(f"📦 Creating CAT-Net architecture and loading weights from: {model_path}")
            
            # Create the model architecture
            self.model = self.create_catnet()
            
            # Load the trained weights
            self.model.load_weights(model_path)
            
            print("✅ CAT-Net model loaded successfully!")
            print(f"   Model name: {self.model.name}")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            print(f"   Total parameters: {self.model.count_params():,}")
            
        except Exception as e:
            raise Exception(f"Error loading CAT-Net model: {e}")
        
        # Load scaler parameters
        scaler_paths = [
            os.path.join('..', 'models', 'scaler_params.json'),
            os.path.join('..', 'frontend', 'public', 'models', 'scaler_params.json')
        ]
        
        scaler_loaded = False
        for scaler_path in scaler_paths:
            if os.path.exists(scaler_path):
                print(f"📊 Loading scaler parameters from: {scaler_path}")
                with open(scaler_path, 'r') as f:
                    self.scaler = json.load(f)
                print("✅ Scaler parameters loaded successfully!")
                print(f"   Mean range: [{np.min(self.scaler['mean']):.6f}, {np.max(self.scaler['mean']):.6f}]")
                print(f"   Scale range: [{np.min(self.scaler['scale']):.6f}, {np.max(self.scaler['scale']):.6f}]")
                scaler_loaded = True
                break
        
        if not scaler_loaded:
            raise FileNotFoundError("Scaler parameters not found!")
            
    def load_patient_data(self):
        """Load the 47 patients hospital data"""
        print("\n🔄 Loading Patient Data...")
        print("=" * 50)
        
        data_path = os.path.join('..', 'data', '47_Patients_Hospital.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Patient data not found at {data_path}")
            
        print(f"📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check if we have patient ID column
        patient_cols = [col for col in df.columns if 'patient' in col.lower() or 'id' in col.lower()]
        if patient_cols:
            print(f"   Patient ID column: {patient_cols[0]}")
        
        return df
    
    def preprocess_ecg_window(self, ecg_data):
        """Preprocess a single ECG window using saved scaler parameters"""
        try:
            # Ensure we have exactly 187 samples
            if len(ecg_data) != 187:
                if len(ecg_data) > 187:
                    ecg_data = ecg_data[:187]
                else:
                    # Pad with zeros
                    ecg_data = list(ecg_data) + [0.0] * (187 - len(ecg_data))
            
            # Standardize using saved scaler parameters
            normalized = np.array([
                (float(value) - self.scaler['mean'][i]) / max(self.scaler['scale'][i], 1e-8)
                for i, value in enumerate(ecg_data)
            ])
            
            return normalized
            
        except Exception as e:
            print(f"⚠️ Error preprocessing ECG data: {e}")
            return np.zeros(187)
    
    def classify_window(self, ecg_window):
        """Classify a single ECG window"""
        try:
            # Preprocess the window
            preprocessed = self.preprocess_ecg_window(ecg_window)
            
            # Reshape for model input (batch, timesteps, features)
            input_tensor = preprocessed.reshape(1, 187, 1)
            
            # Get prediction
            prediction = self.model.predict(input_tensor, verbose=0)
            probabilities = prediction[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            return {
                'class': int(predicted_class),
                'class_name': self.class_names[predicted_class],
                'class_short': self.class_names_short[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            print(f"⚠️ Error classifying window: {e}")
            return {
                'class': 0,
                'class_name': 'Normal',
                'class_short': 'N',
                'confidence': 0.0,
                'probabilities': [1.0, 0.0, 0.0, 0.0, 0.0]
            }
    
    def process_patient_data(self, df):
        """Process all patients and generate classification results"""
        print("\n🔄 Processing Patient Data...")
        print("=" * 50)
        
        # Identify patient ID column and ECG data columns
        patient_id_col = None
        for col in df.columns:
            if 'patient' in col.lower() or 'id' in col.lower():
                patient_id_col = col
                break
        
        if patient_id_col is None:
            print("⚠️ No patient ID column found, using row index as patient ID")
            patient_ids = [f"Patient_{i+1}" for i in range(len(df))]
        else:
            patient_ids = df[patient_id_col].tolist()
        
        # Get ECG data columns (excluding patient ID)
        ecg_columns = [col for col in df.columns if col != patient_id_col]
        print(f"📊 Found {len(patient_ids)} patients")
        print(f"📊 ECG data columns: {len(ecg_columns)}")
        
        # Process each patient
        results = []
        
        for i, patient_id in enumerate(patient_ids):
            print(f"🔄 Processing Patient {patient_id} ({i+1}/{len(patient_ids)})...")
            
            # Get patient's ECG data
            if patient_id_col:
                patient_data = df[df[patient_id_col] == patient_id][ecg_columns].iloc[0].values
            else:
                patient_data = df.iloc[i][ecg_columns].values
            
            # Convert to numeric and handle NaN values
            patient_data = pd.to_numeric(patient_data, errors='coerce')
            patient_data = np.nan_to_num(patient_data, nan=0.0)
            
            # Create windows of 187 values
            window_size = 187
            windows = []
            
            for start_idx in range(0, len(patient_data), window_size):
                end_idx = start_idx + window_size
                window = patient_data[start_idx:end_idx]
                
                if len(window) > 0:  # Only process non-empty windows
                    windows.append(window)
                
                # Stop after 100 windows as requested
                if len(windows) >= 100:
                    break
            
            print(f"   Created {len(windows)} windows for Patient {patient_id}")
            
            # Classify each window
            patient_results = []
            for j, window in enumerate(windows):
                classification = self.classify_window(window)
                patient_results.append({
                    'patient_id': patient_id,
                    'heartbeat': f'heartbeat_{j+1}',
                    'window_start': j * window_size,
                    'window_end': min((j+1) * window_size - 1, len(patient_data) - 1),
                    'classification': classification['class'],
                    'class_name': classification['class_name'],
                    'class_short': classification['class_short'],
                    'confidence': classification['confidence']
                })
            
            results.extend(patient_results)
            
            # Print summary for this patient
            classifications = [r['class_short'] for r in patient_results]
            class_counts = {cls: classifications.count(cls) for cls in set(classifications)}
            print(f"   Classifications: {class_counts}")
        
        return results
    
    def generate_report_table(self, results):
        """Generate the report table as requested"""
        print("\n🔄 Generating Report Table...")
        print("=" * 50)
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Create pivot table with patients as columns and heartbeats as rows
        pivot_table = df_results.pivot(index='heartbeat', columns='patient_id', values='class_short')
        
        # Fill NaN with empty string
        pivot_table = pivot_table.fillna('')
        
        print(f"✅ Report table generated!")
        print(f"   Shape: {pivot_table.shape}")
        print(f"   Patients: {len(pivot_table.columns)}")
        print(f"   Heartbeats: {len(pivot_table.index)}")
        
        return pivot_table
    
    def save_results(self, pivot_table, detailed_results):
        """Save results to files"""
        print("\n🔄 Saving Results...")
        print("=" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save pivot table (main report)
        report_file = f"cat_net_classification_report_{timestamp}.csv"
        pivot_table.to_csv(report_file)
        print(f"📊 Main report saved to: {report_file}")
        
        # Save detailed results
        detailed_file = f"cat_net_detailed_results_{timestamp}.csv"
        df_detailed = pd.DataFrame(detailed_results)
        df_detailed.to_csv(detailed_file, index=False)
        print(f"📋 Detailed results saved to: {detailed_file}")
        
        # Save summary statistics
        summary_file = f"cat_net_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("CAT-Net Classification Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Model: CAT-Net.h5\n")
            f.write(f"Window Size: 187 samples\n")
            f.write(f"Total Patients: {len(pivot_table.columns)}\n")
            f.write(f"Total Heartbeats: {len(pivot_table.index)}\n\n")
            
            # Overall classification distribution
            all_classifications = []
            for col in pivot_table.columns:
                all_classifications.extend([v for v in pivot_table[col].values if v != ''])
            
            f.write("Overall Classification Distribution:\n")
            f.write("-" * 30 + "\n")
            for class_short, class_name in self.class_names_short.items():
                count = all_classifications.count(self.class_names_short[class_short])
                percentage = (count / len(all_classifications)) * 100 if all_classifications else 0
                f.write(f"{class_name} ({self.class_names_short[class_short]}): {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nTotal Classifications: {len(all_classifications)}\n")
        
        print(f"📈 Summary saved to: {summary_file}")
        
        return report_file, detailed_file, summary_file
    
    def run_test(self):
        """Run the complete static test"""
        print("🚀 Starting Static CAT-Net Model Test")
        print("=" * 60)
        print(f"Start Time: {datetime.now().isoformat()}")
        print()
        
        try:
            # Load model and scaler
            self.load_model_and_scaler()
            
            # Load patient data
            df = self.load_patient_data()
            
            # Process all patients
            results = self.process_patient_data(df)
            
            # Generate report table
            pivot_table = self.generate_report_table(results)
            
            # Save results
            report_file, detailed_file, summary_file = self.save_results(pivot_table, results)
            
            # Display sample of results
            print("\n🔍 Sample Results Preview:")
            print("=" * 50)
            print("First 10 rows and 5 columns of the report table:")
            print(pivot_table.iloc[:10, :5])
            
            print(f"\n✅ Static Test Completed Successfully!")
            print(f"📊 Main Report: {report_file}")
            print(f"📋 Detailed Results: {detailed_file}")
            print(f"📈 Summary: {summary_file}")
            print(f"⏱️ End Time: {datetime.now().isoformat()}")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            raise e

if __name__ == "__main__":
    # Run the static test
    tester = StaticCATNetTester()
    tester.run_test() 
#!/usr/bin/env python3
"""
Extract and save scaler parameters for ECG preprocessing.
This script loads the training data, fits a StandardScaler, and saves the parameters.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_scaler_parameters():
    """Extract and save scaler parameters from training data"""
    
    # Configuration
    DATA_DIR = os.path.join('..', 'data')
    MODELS_DIR = '../models'
    OUTPUT_DIR = os.path.join('..', 'frontend', 'public', 'models')
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🔄 Extracting scaler parameters from training data...")
    
    try:
        # Look for training data file
        train_files = [
            'mitbih_train.csv',
            'train.csv',
            'training_data.csv'
        ]
        
        train_path = None
        for filename in train_files:
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                train_path = filepath
                break
        
        if not train_path:
            print("⚠️ Training data file not found. Creating default scaler parameters.")
            # Create default scaler parameters (no scaling)
            scaler_params = {
                'mean': [0.0] * 187,
                'scale': [1.0] * 187
            }
        else:
            print(f"📊 Loading training data from: {train_path}")
            
            # Load training data
            train_df = pd.read_csv(train_path, header=None)
            print(f"Training data shape: {train_df.shape}")
            
            # Extract features (first 187 columns) and labels (last column)
            X_train = train_df.iloc[:, :187].values
            y_train = train_df.iloc[:, 187].values.astype(int)
            
            print(f"Features shape: {X_train.shape}")
            print(f"Labels shape: {y_train.shape}")
            print(f"Label classes: {np.unique(y_train)}")
            
            # Fit StandardScaler
            print("🔧 Fitting StandardScaler...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Extract scaler parameters
            scaler_params = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
            
            print(f"✅ Scaler fitted successfully!")
            print(f"   Mean range: [{np.min(scaler.mean_):.6f}, {np.max(scaler.mean_):.6f}]")
            print(f"   Scale range: [{np.min(scaler.scale_):.6f}, {np.max(scaler.scale_):.6f}]")
            
            # Verify scaling
            scaled_mean = np.mean(X_train_scaled)
            scaled_std = np.std(X_train_scaled)
            print(f"   Scaled data - Mean: {scaled_mean:.6f}, Std: {scaled_std:.6f}")
        
        # Save scaler parameters
        output_path = os.path.join(OUTPUT_DIR, 'scaler_params.json')
        with open(output_path, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"💾 Scaler parameters saved to: {output_path}")
        
        # Also save to models directory for backup
        backup_path = os.path.join(MODELS_DIR, 'scaler_params.json')
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(backup_path, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"💾 Backup saved to: {backup_path}")
        
        return scaler_params
        
    except Exception as e:
        print(f"❌ Error extracting scaler parameters: {e}")
        
        # Create default scaler parameters as fallback
        print("🔄 Creating default scaler parameters...")
        scaler_params = {
            'mean': [0.0] * 187,
            'scale': [1.0] * 187
        }
        
        output_path = os.path.join(OUTPUT_DIR, 'scaler_params.json')
        with open(output_path, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"💾 Default scaler parameters saved to: {output_path}")
        return scaler_params

if __name__ == "__main__":
    print("🚀 ECG Scaler Parameter Extraction")
    print("=" * 50)
    
    scaler_params = extract_scaler_parameters()
    
    print("\n✅ Process completed successfully!")
    print(f"   Scaler parameters have {len(scaler_params['mean'])} features")
    print(f"   Mean values: {len([x for x in scaler_params['mean'] if x != 0.0])} non-zero")
    print(f"   Scale values: {len([x for x in scaler_params['scale'] if x != 1.0])} non-unit") 
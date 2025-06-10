#!/usr/bin/env python3
"""
Debug script to identify CAT-Net model loading issues
"""

import os
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Let's try loading the model without any custom objects first
model_path = os.path.join('..', 'models', 'CAT-Net.h5')

print("Attempting to load CAT-Net model without custom objects...")
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully without custom objects!")
    print(f"Model name: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Failed to load without custom objects: {e}")

# Try inspecting the HDF5 file structure
try:
    import h5py
    print("\n📊 Inspecting HDF5 file structure...")
    with h5py.File(model_path, 'r') as f:
        def print_structure(name, obj):
            print(name)
        f.visititems(print_structure)
except Exception as e:
    print(f"⚠️ Could not inspect HDF5 structure: {e}")

# Try loading with just the ChannelAttention layer
print("\n🔧 Trying with just ChannelAttention...")

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.shared_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=True),
            tf.keras.layers.Dense(filters, kernel_initializer="he_normal", use_bias=True)
        ])
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs):
        avg = self.avg_pool(inputs)
        max_val = self.max_pool(inputs)
        avg = self.shared_mlp(avg)
        max_val = self.shared_mlp(max_val)
        attn = self.sigmoid(avg + max_val)[:, None, :]
        return inputs * attn
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio
        })
        return config

try:
    custom_objects = {'ChannelAttention': ChannelAttention}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print("✅ Model loaded successfully with ChannelAttention!")
    print(f"Model name: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Print model summary
    print("\n📋 Model Summary:")
    model.summary()
    
except Exception as e:
    print(f"❌ Failed to load with ChannelAttention: {e}")

# Try loading with compile=False
print("\n🔧 Trying with compile=False...")
try:
    custom_objects = {'ChannelAttention': ChannelAttention}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✅ Model loaded successfully with compile=False!")
    print(f"Model name: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test a prediction
    print("\n🧪 Testing prediction...")
    import numpy as np
    test_input = np.random.randn(1, 187, 1)
    prediction = model.predict(test_input, verbose=0)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction: {prediction}")
    
except Exception as e:
    print(f"❌ Failed to load with compile=False: {e}")

# Try loading weights only and recreating the model architecture
print("\n🔧 Trying to load weights only...")
try:
    # Let's try to inspect the JSON config first
    with h5py.File(model_path, 'r') as f:
        if 'model_config' in f.attrs:
            config = f.attrs['model_config']
            print(f"Model config found: {len(str(config))} characters")
            # Try to decode and print part of the config
            import json
            try:
                config_dict = json.loads(config)
                print(f"Config class name: {config_dict.get('class_name', 'Unknown')}")
                print(f"Config keys: {list(config_dict.keys())}")
            except:
                print("Could not parse config as JSON")
        else:
            print("No model config found in HDF5 file")
            
except Exception as e:
    print(f"❌ Failed to inspect model config: {e}") 
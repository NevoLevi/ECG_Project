#!/usr/bin/env python3
"""
Working CAT-Net Model Loader
"""

import os
import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Flexible ChannelAttention layer that can handle various call signatures
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters=None, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        # Make filters optional with a default
        self.filters = filters if filters is not None else 64
        self.ratio = ratio
        
    def build(self, input_shape):
        # If filters wasn't provided, infer from input shape
        if hasattr(input_shape, '__iter__') and len(input_shape) > 2:
            self.filters = input_shape[-1]
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        
        # Handle the case where filters//ratio might be 0
        mlp_units = max(1, self.filters // self.ratio)
        
        self.shared_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_units, activation="relu", kernel_initializer="he_normal", use_bias=True),
            tf.keras.layers.Dense(self.filters, kernel_initializer="he_normal", use_bias=True)
        ])
        self.sigmoid = tf.keras.layers.Activation("sigmoid")
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: (batch, time, channels)
        avg = self.avg_pool(inputs)
        max_val = self.max_pool(inputs)
        # shared MLP expects 2D
        avg = self.shared_mlp(avg)
        max_val = self.shared_mlp(max_val)
        attn = self.sigmoid(avg + max_val)[:, None, :]    # (batch, 1, channels)
        return inputs * attn
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio
        })
        return config

# Alternative simpler ChannelAttention
class ChannelAttentionSimple(tf.keras.layers.Layer):
    def __init__(self, filters=None, ratio=8, **kwargs):
        super(ChannelAttentionSimple, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        
    def build(self, input_shape):
        if self.filters is None:
            self.filters = input_shape[-1]
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        mlp_units = max(1, self.filters // self.ratio)
        self.shared_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_units, activation="relu"),
            tf.keras.layers.Dense(self.filters)
        ])
        self.sigmoid = tf.keras.layers.Activation("sigmoid")
        super(ChannelAttentionSimple, self).build(input_shape)

    def call(self, inputs):
        avg = self.avg_pool(inputs)
        max_val = self.max_pool(inputs)
        avg = self.shared_mlp(avg)
        max_val = self.shared_mlp(max_val)
        attn = self.sigmoid(avg + max_val)[:, None, :]
        return inputs * attn
    
    def get_config(self):
        config = super(ChannelAttentionSimple, self).get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio
        })
        return config

def try_load_model():
    model_path = os.path.join('..', 'models', 'CAT-Net.h5')
    
    # Try different custom object configurations
    configurations = [
        {"ChannelAttention": ChannelAttention},
        {"ChannelAttention": ChannelAttentionSimple},
        {"ChannelAttention": ChannelAttention, "ChannelAttentionSimple": ChannelAttentionSimple},
    ]
    
    for i, custom_objects in enumerate(configurations):
        print(f"\n🔧 Trying configuration {i+1}...")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print("✅ SUCCESS! Model loaded!")
            print(f"   Model name: {model.name}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Total parameters: {model.count_params():,}")
            
            # Test prediction
            print("\n🧪 Testing prediction...")
            test_input = np.random.randn(1, 187, 1)
            prediction = model.predict(test_input, verbose=0)
            print(f"   Prediction shape: {prediction.shape}")
            print(f"   Prediction values: {prediction[0]}")
            print(f"   Predicted class: {np.argmax(prediction[0])}")
            
            return model
            
        except Exception as e:
            print(f"❌ Configuration {i+1} failed: {e}")
    
    return None

if __name__ == "__main__":
    print("🚀 Attempting to load CAT-Net model...")
    model = try_load_model()
    
    if model is not None:
        print("\n✅ Model successfully loaded and tested!")
    else:
        print("\n❌ All configurations failed") 
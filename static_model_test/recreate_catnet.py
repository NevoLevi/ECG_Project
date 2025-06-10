#!/usr/bin/env python3
"""
Recreate CAT-Net and load weights
"""

import os
import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def transformer_encoder_block(x, embed_dim, num_heads, ff_dim):
    # Multi-Head Self-Attention
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    # Feed-Forward
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn_output = tf.keras.layers.Dense(embed_dim)(ffn_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

def create_catnet(input_shape=(187,1), num_classes=5):
    """
    Recreate the exact CAT-Net architecture from the notebook
    """
    inputs = tf.keras.layers.Input(shape=input_shape)   # (187,1)
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

def load_weights_from_saved_model():
    """
    Create a new CAT-Net and try to load weights from the saved model
    """
    model_path = os.path.join('..', 'models', 'CAT-Net.h5')
    
    print("🔨 Creating new CAT-Net model...")
    try:
        # Create fresh model
        new_model = create_catnet()
        print("✅ New CAT-Net model created successfully!")
        print(f"   Input shape: {new_model.input_shape}")
        print(f"   Output shape: {new_model.output_shape}")
        print(f"   Total parameters: {new_model.count_params():,}")
        
        # Try to load weights
        print("\n📦 Attempting to load weights from saved model...")
        new_model.load_weights(model_path)
        print("✅ Weights loaded successfully!")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        test_input = np.random.randn(1, 187, 1)
        prediction = new_model.predict(test_input, verbose=0)
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Prediction values: {prediction[0]}")
        print(f"   Predicted class: {np.argmax(prediction[0])}")
        
        return new_model
        
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return None

def use_saved_model_direct():
    """
    Try to use the saved model directly but with different custom object approaches
    """
    model_path = os.path.join('..', 'models', 'CAT-Net.h5')
    
    # Simple ChannelAttention that only accepts kwargs
    class SimpleChannelAttention(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            # Extract filters if provided
            filters = kwargs.pop('filters', 64)
            ratio = kwargs.pop('ratio', 8)
            super().__init__(**kwargs)
            self.filters = filters
            self.ratio = ratio
            
        def build(self, input_shape):
            self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
            self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
            self.shared_mlp = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.filters // self.ratio, activation="relu"),
                tf.keras.layers.Dense(self.filters)
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
            config = super().get_config()
            config.update({'filters': self.filters, 'ratio': self.ratio})
            return config
    
    print("🔧 Trying to load saved model with simple custom objects...")
    try:
        custom_objects = {'ChannelAttention': SimpleChannelAttention}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✅ Saved model loaded successfully!")
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
        print(f"❌ Failed to load saved model: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Attempting to load your CAT-Net model...")
    
    # First try the direct approach
    model = use_saved_model_direct()
    
    # If that fails, try recreating and loading weights
    if model is None:
        print("\n" + "="*60)
        print("Direct loading failed, trying weight loading approach...")
        model = load_weights_from_saved_model()
    
    if model is not None:
        print(f"\n✅ SUCCESS! Your actual CAT-Net model is now loaded and working!")
        print("📋 Model Summary:")
        model.summary()
    else:
        print(f"\n❌ Unable to load the model with any approach") 
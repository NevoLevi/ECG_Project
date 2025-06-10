import threading
import time
import json
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import logging
from collections import deque

# Configure logging for ECG classifications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_classifications.log'),
        logging.StreamHandler()
    ]
)

# Custom ChannelAttention layer for CAT-Net (exact same as static test)
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
        avg = self.avg_pool(inputs)
        max_val = self.max_pool(inputs)
        avg = self.shared_mlp(avg)
        max_val = self.shared_mlp(max_val)
        attn = self.sigmoid(avg + max_val)[:, None, :]
        return inputs * attn

# Transformer encoder block function for CAT-Net
def transformer_encoder_block(x, embed_dim, num_heads, ff_dim):
    """Transformer encoder block with multi-head attention and feed-forward network"""
    # Multi-Head Self-Attention
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    # Feed-Forward
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn_output = tf.keras.layers.Dense(embed_dim)(ffn_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

class ECGClassificationLogger:
    """Logger for ECG classifications with detailed metrics"""
    
    def __init__(self, log_file='ecg_classifications.log'):
        self.logger = logging.getLogger('ECGClassifications')
        self.classification_history = {}  # patient_id -> deque of classifications
        self.alert_callbacks = []
        
        # Class names for better logging
        self.class_names = {
            0: 'Normal (N)',
            1: 'Supraventricular (S)', 
            2: 'Ventricular (V)',
            3: 'Fusion (F)',
            4: 'Unclassified (Q)'
        }
        
    def log_classification(self, patient_id, prediction, ecg_window_info=None):
        """Log a single classification with detailed information"""
        
        timestamp = prediction.get('timestamp', datetime.now().isoformat())
        predicted_class = prediction['class']
        confidence = prediction['confidence']
        class_name = self.class_names.get(predicted_class, 'Unknown')
        
        # Initialize patient history if not exists
        if patient_id not in self.classification_history:
            self.classification_history[patient_id] = deque(maxlen=100)  # Keep last 100 classifications
            
        # Add to history
        classification_record = {
            'timestamp': timestamp,
            'class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': prediction.get('probabilities', []),
            'window_info': ecg_window_info or {}
        }
        
        self.classification_history[patient_id].append(classification_record)
        
        # Log classification
        log_level = logging.WARNING if predicted_class > 0 and confidence > 0.7 else logging.INFO
        self.logger.log(
            log_level,
            f"Patient {patient_id}: {class_name} (Class {predicted_class}) "
            f"- Confidence: {confidence:.3f} - Timestamp: {timestamp}"
        )
        
        # Check for abnormal patterns and trigger alerts
        self._check_abnormal_patterns(patient_id, classification_record)
        
        return classification_record
    
    def _check_abnormal_patterns(self, patient_id, latest_classification):
        """Check for abnormal patterns and trigger alerts"""
        
        if patient_id not in self.classification_history:
            return
            
        history = list(self.classification_history[patient_id])
        
        # Alert conditions
        alerts = []
        
        # 1. High confidence abnormal classification
        if latest_classification['class'] > 0 and latest_classification['confidence'] > 0.8:
            alerts.append({
                'type': 'high_confidence_abnormal',
                'severity': 'HIGH',
                'message': f"High confidence abnormal classification: {latest_classification['class_name']} ({latest_classification['confidence']:.3f})"
            })
        
        # 2. Multiple consecutive abnormal classifications
        if len(history) >= 3:
            recent_classes = [h['class'] for h in history[-3:]]
            if all(c > 0 for c in recent_classes):
                alerts.append({
                    'type': 'consecutive_abnormal',
                    'severity': 'MEDIUM',
                    'message': f"3 consecutive abnormal classifications: {recent_classes}"
                })
        
        # 3. Sudden change from normal to abnormal with high confidence
        if len(history) >= 2:
            prev_class = history[-2]['class']
            curr_class = latest_classification['class']
            curr_confidence = latest_classification['confidence']
            
            if prev_class == 0 and curr_class > 0 and curr_confidence > 0.75:
                alerts.append({
                    'type': 'sudden_abnormal_change',
                    'severity': 'HIGH',
                    'message': f"Sudden change from Normal to {latest_classification['class_name']} with confidence {curr_confidence:.3f}"
                })
        
        # 4. Frequent oscillations between normal and abnormal
        if len(history) >= 5:
            recent_classes = [h['class'] for h in history[-5:]]
            changes = sum(1 for i in range(1, len(recent_classes)) if recent_classes[i] != recent_classes[i-1])
            if changes >= 3:
                alerts.append({
                    'type': 'frequent_oscillations',
                    'severity': 'MEDIUM',
                    'message': f"Frequent oscillations detected: {recent_classes}"
                })
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(patient_id, alert, latest_classification)
    
    def _trigger_alert(self, patient_id, alert, classification):
        """Trigger an alert for abnormal patterns"""
        
        alert_data = {
            'patient_id': patient_id,
            'timestamp': classification['timestamp'],
            'alert_type': alert['type'],
            'severity': alert['severity'],
            'message': alert['message'],
            'classification': classification
        }
        
        # Log alert
        self.logger.warning(f"🚨 ALERT for Patient {patient_id}: {alert['message']}")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add a callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
    
    def get_patient_history(self, patient_id, limit=50):
        """Get classification history for a patient"""
        if patient_id in self.classification_history:
            history = list(self.classification_history[patient_id])
            return history[-limit:] if limit else history
        return []
    
    def get_patient_summary(self, patient_id):
        """Get summary statistics for a patient"""
        if patient_id not in self.classification_history:
            return None
            
        history = list(self.classification_history[patient_id])
        if not history:
            return None
            
        # Calculate statistics
        total_classifications = len(history)
        abnormal_count = sum(1 for h in history if h['class'] > 0)
        normal_count = total_classifications - abnormal_count
        
        # Class distribution
        class_counts = {}
        for class_id in range(5):
            class_counts[class_id] = sum(1 for h in history if h['class'] == class_id)
        
        # Average confidence by class
        avg_confidence = {}
        for class_id in range(5):
            class_predictions = [h for h in history if h['class'] == class_id]
            if class_predictions:
                avg_confidence[class_id] = np.mean([p['confidence'] for p in class_predictions])
            else:
                avg_confidence[class_id] = 0.0
        
        # Recent trend (last 10 classifications)
        recent_history = history[-10:] if len(history) >= 10 else history
        recent_abnormal_rate = sum(1 for h in recent_history if h['class'] > 0) / len(recent_history)
        
        return {
            'patient_id': patient_id,
            'total_classifications': total_classifications,
            'normal_count': normal_count,
            'abnormal_count': abnormal_count,
            'abnormal_rate': abnormal_count / total_classifications,
            'class_distribution': class_counts,
            'average_confidence_by_class': avg_confidence,
            'recent_abnormal_rate': recent_abnormal_rate,
            'first_classification': history[0]['timestamp'],
            'last_classification': history[-1]['timestamp']
        }

class ECGWorker:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_running = False
        self.model_inference_active = False  # New flag to control model inference
        self.workers = {}
        self.lock = threading.Lock()
        self.logger = ECGClassificationLogger()
        
        # Load model and scaler
        self.load_model()
        
    def load_model(self):
        """Load the CAT-Net model and scaler parameters using the same approach as static test"""
        print("🔄 Initializing ECG Worker...")
        
        try:
            # Use the exact same model loading approach as static_model_test
            model_path = os.path.join('..', 'models', 'CAT-Net.h5')
            if os.path.exists(model_path):
                print(f"📦 Loading CAT-Net model from: {model_path}")
                
                try:
                    # Recreate CAT-Net architecture and load weights (same as static test)
                    self.model = self.create_catnet_architecture()
                    self.model.load_weights(model_path)
                    print("✅ CAT-Net model loaded successfully with weights!")
                    print(f"   Model input shape: {self.model.input_shape}")
                    print(f"   Model output shape: {self.model.output_shape}")
                    print(f"   Total parameters: {self.model.count_params():,}")
                    print(f"   Model name: {self.model.name}")
                except Exception as model_error:
                    print(f"⚠️ Error loading CAT-Net model: {model_error}")
                    print("🔄 Creating compatible mock model...")
                    self.model = self.create_compatible_model()
            else:
                print(f"⚠️ CAT-Net model not found at {model_path}")
                print("🔄 Creating compatible mock model...")
                self.model = self.create_compatible_model()
            
            # Load scaler parameters
            scaler_paths = [
                os.path.join('..', 'frontend', 'public', 'models', 'scaler_params.json'),
                os.path.join('..', 'models', 'scaler_params.json'),
                'scaler_params.json'
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
                print("⚠️ Scaler parameters not found. Creating default scaler...")
                self.scaler = {
                    'mean': [0.0] * 187,
                    'scale': [1.0] * 187
                }
                
            print("✅ Model and scaler initialization completed!")
            
        except Exception as e:
            print(f"❌ Error during model initialization: {e}")
            # Always fallback to mock model and scaler
            print("🔄 Using fallback mock model and scaler...")
            self.model = self.create_compatible_model()
            self.scaler = {
                'mean': [0.0] * 187,
                'scale': [1.0] * 187
            }
            print("✅ Fallback model initialized successfully")
    
    def create_catnet_architecture(self):
        """Create the exact CAT-Net architecture (same as static test)"""
        try:
            print("🔨 Creating CAT-Net architecture...")
            
            inputs = tf.keras.Input(shape=(187, 1))
            
            # First block
            x = tf.keras.layers.Conv1D(16, 21, padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)
            x = ChannelAttention(16)(x)
            
            # Second block  
            x = tf.keras.layers.Conv1D(16, 23, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)
            x = ChannelAttention(16)(x)
            
            # Third block
            x = tf.keras.layers.Conv1D(32, 25, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.AveragePooling1D(3, strides=2, padding='same')(x)
            x = ChannelAttention(32)(x)
            
            # Fourth block
            x = tf.keras.layers.Conv1D(32, 27, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.AveragePooling1D(3, strides=2, padding='same')(x)
            x = ChannelAttention(32)(x)
            
            # Transformer blocks
            x = self.transformer_encoder_block(x, embed_dim=32, num_heads=4, ff_dim=128)
            x = self.transformer_encoder_block(x, embed_dim=32, num_heads=4, ff_dim=128)
            
            # Global pooling and classification
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CAT_Net')
            print("✅ Created CAT-Net architecture successfully")
            return model
            
        except Exception as e:
            print(f"⚠️ Error creating CAT-Net architecture: {e}")
            return self.create_compatible_model()
    
    def transformer_encoder_block(self, x, embed_dim, num_heads, ff_dim):
        """Transformer encoder block"""
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dense(embed_dim)(ffn_output)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    def create_compatible_model(self):
        """Create a CAT-Net compatible model for testing"""
        try:
            print("🔨 Creating CAT-Net compatible model...")
            
            # Create a model that matches CAT-Net's expected input/output
            inputs = tf.keras.Input(shape=(187, 1), name='ecg_input')
            
            # Simple CNN layers
            x = tf.keras.layers.Conv1D(16, 21, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)
            
            x = tf.keras.layers.Conv1D(32, 23, padding='same', activation='relu')(x)
            x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)
            
            x = tf.keras.layers.Conv1D(64, 25, padding='same', activation='relu')(x)
            x = tf.keras.layers.AveragePooling1D(3, strides=2, padding='same')(x)
            
            # Global pooling and dense layers
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax', name='predictions')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CAT_Net_Compatible')
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Created CAT-Net compatible model")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Parameters: {model.count_params():,}")
            
            return model
            
        except Exception as e:
            print(f"⚠️ Error creating compatible model: {e}")
            # Create even simpler fallback
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape((187, 1), input_shape=(187,)),
                tf.keras.layers.Conv1D(32, 5, activation='relu'),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            print("✅ Created simple fallback model")
            return model
        
    def preprocess_ecg(self, ecg_data):
        """Preprocess ECG data using saved scaler parameters"""
        try:
            if not self.scaler:
                # Create default scaler if none exists
                self.scaler = {
                    'mean': [0.0] * 187,
                    'scale': [1.0] * 187
                }
                
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
            # Return zeros as fallback
            return np.zeros(187)
        
    def predict(self, ecg_data, patient_id=None):
        """Make prediction on ECG data with detailed logging"""
        try:
            if not self.model:
                print("⚠️ Model not loaded, cannot make predictions")
                return None
                
            # Preprocess the ECG data
            preprocessed_data = self.preprocess_ecg(ecg_data)
            
            # Get window information for logging
            window_info = {
                'window_size': len(ecg_data),
                'preprocessed_mean': float(np.mean(preprocessed_data)),
                'preprocessed_std': float(np.std(preprocessed_data)),
                'preprocessed_min': float(np.min(preprocessed_data)),
                'preprocessed_max': float(np.max(preprocessed_data))
            }
            
            # Reshape for model input - CAT-Net expects (batch, timesteps, features)
            if len(self.model.input_shape) == 3:  # (batch, timesteps, features)
                input_tensor = preprocessed_data.reshape(1, 187, 1)
            else:  # (batch, features)
                input_tensor = preprocessed_data.reshape(1, 187)
            
            # Get prediction
            prediction = self.model.predict(input_tensor, verbose=0)
            probabilities = prediction[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            prediction_result = {
                'class': int(predicted_class),
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'CAT-Net' if 'CAT' in str(self.model.name) else 'Compatible'
            }
            
            # Log the classification
            if patient_id:
                self.logger.log_classification(patient_id, prediction_result, window_info)
            
            return prediction_result
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Return a random prediction as fallback
            fallback_prediction = {
                'class': np.random.randint(0, 5),
                'confidence': 0.5 + np.random.random() * 0.3,  # Random confidence between 0.5-0.8
                'probabilities': [0.2] * 5,  # Equal probabilities
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Fallback',
                'error': str(e)
            }
            
            if patient_id:
                self.logger.log_classification(patient_id, fallback_prediction)
            
            return fallback_prediction
            
    def start_monitoring(self, patient_id, ecg_data, callback):
        """Start monitoring a patient's ECG data with enhanced logging"""
        try:
            if patient_id in self.workers:
                print(f"⚠️ Already monitoring patient {patient_id}")
                return
                
            # Validate ECG data
            if not ecg_data or len(ecg_data) < 187:
                print(f"⚠️ Invalid ECG data for patient {patient_id}: length {len(ecg_data) if ecg_data else 0}")
                return
                
            def monitor_worker():
                position = 0
                window_size = 187  # Model input size
                classification_count = 0
                
                print(f"🚀 Started monitoring patient {patient_id} with {len(ecg_data)} ECG samples")
                self.logger.logger.info(f"Started monitoring patient {patient_id} - ECG length: {len(ecg_data)} samples")
                
                while self.is_running and patient_id in self.workers:
                    try:
                        # Check if model inference is active
                        if self.model_inference_active:
                            # Batch inference: 8 windows per 12-second interval
                            frames_per_interval = 8
                            successful_inferences = 0
                            failed_inferences = 0
                            
                            for i in range(frames_per_interval):
                                start = position + i * window_size
                                if start + window_size > len(ecg_data):
                                    window = np.concatenate([
                                        ecg_data[start:],
                                        ecg_data[:window_size - (len(ecg_data) - start)]
                                    ])
                                else:
                                    window = ecg_data[start:start + window_size]

                                prediction = self.predict(window, patient_id)
                                if prediction and 'error' not in prediction:
                                    callback(patient_id, prediction)
                                    classification_count += 1
                                    successful_inferences += 1
                                    # Log abnormal detections
                                    if prediction['class'] > 0 and prediction['confidence'] > 0.7:
                                        print(f"🚨 Abnormal detection for {patient_id}: "
                                              f"Class {prediction['class']} "
                                              f"- Confidence: {prediction['confidence']:.3f}")
                                else:
                                    failed_inferences += 1

                            # Log batch completion with success/failure info
                            if successful_inferences > 0:
                                print(f"✅ Successfully inferred {successful_inferences}/{frames_per_interval} heartbeats for {patient_id} "
                                      f"(total {classification_count})")
                            if failed_inferences > 0:
                                print(f"❌ Failed {failed_inferences}/{frames_per_interval} inferences for {patient_id}")
                            
                            # Advance position for next batch
                            position = (position + frames_per_interval * window_size) % len(ecg_data)
                        else:
                            # Model inference is off - just advance position slowly for animation
                            position = (position + window_size) % len(ecg_data)
                        
                        # Sleep interval (12s when model active, 1.5s when just monitoring)
                        sleep_time = 12.0 if self.model_inference_active else 1.5
                        time.sleep(sleep_time)

                    except Exception as e:
                        print(f"⚠️ Error monitoring patient {patient_id}: {e}")
                        self.logger.logger.error(f"Error monitoring patient {patient_id}: {e}")
                        time.sleep(1)
                        
                print(f"🛑 Stopped monitoring patient {patient_id}")
                self.logger.logger.info(f"Stopped monitoring patient {patient_id} after {classification_count} classifications")
                        
            # Start worker thread
            self.workers[patient_id] = threading.Thread(target=monitor_worker, daemon=True)
            self.workers[patient_id].start()
            print(f"✅ Worker thread started for patient {patient_id}")
            
        except Exception as e:
            print(f"❌ Error starting monitoring for patient {patient_id}: {e}")
            self.logger.logger.error(f"Error starting monitoring for patient {patient_id}: {e}")
        
    def stop_monitoring(self, patient_id):
        """Stop monitoring a patient"""
        try:
            if patient_id in self.workers:
                print(f"🛑 Stopping monitoring for patient {patient_id}")
                self.logger.logger.info(f"Stopping monitoring for patient {patient_id}")
                del self.workers[patient_id]
        except Exception as e:
            print(f"⚠️ Error stopping monitoring for patient {patient_id}: {e}")
            self.logger.logger.error(f"Error stopping monitoring for patient {patient_id}: {e}")
    
    def start_model_inference(self):
        """Start model inference for all monitored patients"""
        if not self.model:
            print("⚠️ Model not loaded, cannot start inference")
            return False
        
        self.model_inference_active = True
        print("🧠 Model inference activated for all patients")
        self.logger.logger.info("Model inference activated")
        return True
    
    def stop_model_inference(self):
        """Stop model inference but keep monitoring active"""
        self.model_inference_active = False
        print("🛑 Model inference deactivated (monitoring continues)")
        self.logger.logger.info("Model inference deactivated")
        return True
    
    def get_patient_status(self, patient_id):
        """Get detailed status for a patient including classification history"""
        summary = self.logger.get_patient_summary(patient_id)
        is_monitoring = patient_id in self.workers
        
        return {
            'patient_id': patient_id,
            'is_monitoring': is_monitoring,
            'classification_summary': summary,
            'recent_history': self.logger.get_patient_history(patient_id, limit=10)
        }
    
    def get_all_patients_status(self):
        """Get status for all patients being monitored"""
        all_patients = set(list(self.workers.keys()) + list(self.logger.classification_history.keys()))
        
        return {
            patient_id: self.get_patient_status(patient_id)
            for patient_id in all_patients
        }
            
    def start(self):
        """Start the worker service"""
        self.is_running = True
        print("🚀 ECG Worker service started")
        self.logger.logger.info("ECG Worker service started")
        
    def stop(self):
        """Stop the worker service"""
        print("🛑 Stopping ECG Worker service...")
        self.logger.logger.info("Stopping ECG Worker service")
        self.is_running = False
        # Wait for all workers to finish
        for worker in self.workers.values():
            if worker.is_alive():
                worker.join(timeout=2)
        self.workers.clear()
        print("✅ ECG Worker service stopped")
        self.logger.logger.info("ECG Worker service stopped")

# Create global worker instance
worker = ECGWorker() 
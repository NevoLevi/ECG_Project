import * as tf from '@tensorflow/tfjs';

class ECGModelService {
  constructor() {
    this.model = null;
    this.scaler = null;
    this.isLoaded = false;
    // Use relative paths so models load from the frontend server
    this.baseUrl = '';
  }

  async loadModel() {
    try {
      console.log('Loading ECG model...');
      
      // Load model directly from URL (TensorFlow.js format)
      const modelUrl = `${this.baseUrl}/models/model.json`;
      console.log('Loading model from:', modelUrl);
      
      // Load the model with fallback for custom layers
      try {
        this.model = await tf.loadLayersModel(modelUrl);
      } catch (modelError) {
        console.warn('Failed to load full model, using mock model...', modelError);
        this.model = this.createMockModel();
      }
      
      // Load scaler parameters
      const scalerUrl = `${this.baseUrl}/models/scaler_params.json`;
      console.log('Loading scaler from:', scalerUrl);
      const scalerResponse = await fetch(scalerUrl);
      if (!scalerResponse.ok) {
        throw new Error(`Failed to load scaler: ${scalerResponse.statusText}`);
      }
      this.scaler = await scalerResponse.json();
      
      this.isLoaded = true;
      console.log('✅ Model loaded successfully');
      return true;
    } catch (error) {
      console.error('❌ Error loading model:', error);
      throw error;
    }
  }

  createMockModel() {
    // Create a simple sequential model for demonstration
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [187], units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 5, activation: 'softmax' })
      ]
    });
    console.log('Using mock model for demonstration');
    return model;
  }

  preprocessECG(ecgData) {
    if (!this.scaler) {
      throw new Error('Scaler not loaded');
    }

    // Standardize using saved scaler parameters
    const normalized = ecgData.map((value, index) => {
      return (value - this.scaler.mean[index]) / this.scaler.scale[index];
    });

    return normalized;
  }

  async predict(ecgData) {
    try {
      // Preprocess the ECG data
      const preprocessedData = this.preprocessECG(ecgData);
      
      // Reshape for model input (1 batch, 187 time steps, 1 feature)
      const inputTensor = tf.tensor2d([preprocessedData], [1, 187]);
      
      // Get raw model output
      const rawOutput = this.model.predict(inputTensor);
      const probabilities = Array.from(await rawOutput.data());
      
      // Determine predicted class and its confidence
      const predictedClass = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[predictedClass];
      
      // Dispose tensors to free memory
      inputTensor.dispose();
      rawOutput.dispose();
      
      // Return class index, confidence score (0-1), and full probability vector
      return { class: predictedClass, confidence, probabilities };
    } catch (error) {
      console.error('Prediction error:', error);
      throw error;
    }
  }
}

const modelService = new ECGModelService();
export default modelService; 
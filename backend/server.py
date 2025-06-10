from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from worker import worker

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Get the absolute path to the models directory
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'models'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
PATIENT_STATUS_FILE = os.path.join(DATA_DIR, 'patient_status.json')

# Global alerts storage (in production, use a database)
alerts_queue = []

def handle_alert_callback(alert_data):
    """Handle alerts from the worker and store them"""
    alerts_queue.append(alert_data)
    # Keep only the last 1000 alerts (increased from 100)
    if len(alerts_queue) > 1000:
        alerts_queue.pop(0)
    
    print(f"🚨 NEW ALERT: {alert_data['message']} for Patient {alert_data['patient_id']}")

# Register alert callback with worker
worker.logger.add_alert_callback(handle_alert_callback)

# Initialize patient status storage
def load_patient_status():
    """Load patient status from JSON file"""
    if os.path.exists(PATIENT_STATUS_FILE):
        try:
            with open(PATIENT_STATUS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_patient_status(status_data):
    """Save patient status to JSON file"""
    try:
        with open(PATIENT_STATUS_FILE, 'w') as f:
            json.dump(status_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving patient status: {e}")
        return False

def handle_prediction(patient_id, prediction):
    """Handle prediction results from worker with classification counts and status logic."""
    try:
        status_data = load_patient_status()
        
        # Initialize patient record if not exists
        if patient_id not in status_data:
            status_data[patient_id] = {
                'classification_counts': {str(i): 0 for i in range(5)},
                'status': 'N',
                'last_abnormal_detection': None,
                'doctor_checked': False,
                'last_check_time': None,
                'notes': '',
                'last_prediction': None
            }
        
        record = status_data[patient_id]
        # Update classification counts
        class_int = prediction['class']
        class_str = str(class_int)
        record['classification_counts'][class_str] = record['classification_counts'].get(class_str, 0) + 1
        
        # Update last_prediction
        record['last_prediction'] = prediction
        
        # Update last_abnormal_detection if abnormal with high confidence
        if class_int > 0 and prediction.get('confidence', 0) > 0.7:
            record['last_abnormal_detection'] = prediction['timestamp']
            record['doctor_checked'] = False
        
        # Determine status based on counts
        total_abnormal = sum(record['classification_counts'][cls] for cls in ['1', '2', '3', '4'])
        if total_abnormal >= 2:
            abnormal_counts = {cls: record['classification_counts'][cls] for cls in ['1', '2', '3', '4']}
            max_count = max(abnormal_counts.values())
            candidates = [cls for cls, cnt in abnormal_counts.items() if cnt == max_count]
            chosen_cls = min(candidates, key=int)
            code_map = {'1': 'S', '2': 'V', '3': 'F', '4': 'Q'}
            record['status'] = code_map.get(chosen_cls, 'N')
        else:
            record['status'] = 'N'
        
        status_data[patient_id] = record
        save_patient_status(status_data)
        
    except Exception as e:
        print(f"Error handling prediction for patient {patient_id}: {e}")

@app.route('/models/<path:filename>')
def serve_models(filename):
    try:
        return send_from_directory(MODELS_DIR, filename)
    except Exception as e:
        print(f"Error serving {filename}: {str(e)}")
        return {"error": str(e)}, 404

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve data files including the new ECG hospital data"""
    try:
        return send_from_directory(DATA_DIR, filename)
    except Exception as e:
        print(f"Error serving {filename}: {str(e)}")
        return {"error": str(e)}, 404

@app.route('/api/patient-status', methods=['GET'])
def get_patient_status():
    """Get all patient statuses"""
    status_data = load_patient_status()
    return jsonify(status_data)

@app.route('/api/patient-status/<patient_id>', methods=['GET'])
def get_single_patient_status(patient_id):
    """Get status for a specific patient"""
    status_data = load_patient_status()
    patient_status = status_data.get(patient_id, {
        'status': 'normal',
        'last_abnormal_detection': None,
        'doctor_checked': True,
        'last_check_time': None,
        'notes': ''
    })
    return jsonify(patient_status)

@app.route('/api/patient-status/<patient_id>', methods=['POST'])
def update_patient_status(patient_id):
    """Update patient status"""
    try:
        data = request.get_json()
        status_data = load_patient_status()
        
        # Update or create patient status
        if patient_id not in status_data:
            status_data[patient_id] = {
                'status': 'normal',
                'last_abnormal_detection': None,
                'doctor_checked': True,
                'last_check_time': None,
                'notes': ''
            }
        
        # Update fields from request
        if 'status' in data:
            status_data[patient_id]['status'] = data['status']
        
        if 'doctor_checked' in data:
            status_data[patient_id]['doctor_checked'] = data['doctor_checked']
            if data['doctor_checked']:
                status_data[patient_id]['last_check_time'] = datetime.now().isoformat()
        
        if 'notes' in data:
            status_data[patient_id]['notes'] = data['notes']
        
        if 'abnormal_detection' in data:
            status_data[patient_id]['last_abnormal_detection'] = data['abnormal_detection']
            status_data[patient_id]['status'] = 'abnormal'
            status_data[patient_id]['doctor_checked'] = False
        
        # Save to file
        success = save_patient_status(status_data)
        if success:
            return jsonify(status_data[patient_id])
        else:
            return jsonify({'error': 'Failed to save patient status'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/patient-status/<patient_id>/check', methods=['POST'])
def doctor_check_patient(patient_id):
    """Mark patient as checked by doctor"""
    try:
        data = request.get_json()
        status_data = load_patient_status()
        
        if patient_id not in status_data:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Mark as checked
        status_data[patient_id]['doctor_checked'] = True
        status_data[patient_id]['last_check_time'] = datetime.now().isoformat()
        
        # Optionally reset to normal if doctor confirms
        if data.get('reset_to_normal', False):
            status_data[patient_id]['status'] = 'normal'
        
        if 'notes' in data:
            status_data[patient_id]['notes'] = data['notes']
        
        success = save_patient_status(status_data)
        if success:
            return jsonify(status_data[patient_id])
        else:
            return jsonify({'error': 'Failed to save patient status'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# New endpoints for enhanced monitoring

@app.route('/api/monitoring/status', methods=['GET'])
def get_monitoring_status():
    """Get detailed monitoring status for all patients"""
    try:
        return jsonify(worker.get_all_patients_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/patient/<patient_id>', methods=['GET'])
def get_patient_monitoring_status(patient_id):
    """Get detailed monitoring status for a specific patient"""
    try:
        return jsonify(worker.get_patient_status(patient_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/history/<patient_id>', methods=['GET'])
def get_patient_classification_history(patient_id):
    """Get classification history for a patient"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = worker.logger.get_patient_history(patient_id, limit=limit)
        summary = worker.logger.get_patient_summary(patient_id)
        
        return jsonify({
            'patient_id': patient_id,
            'history': history,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all recent alerts"""
    try:
        # Get limit from query parameter
        limit = request.args.get('limit', 50, type=int)
        
        # Return recent alerts (most recent first)
        recent_alerts = alerts_queue[-limit:] if limit else alerts_queue
        recent_alerts.reverse()  # Most recent first
        
        return jsonify({
            'alerts': recent_alerts,
            'total_count': len(alerts_queue)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<patient_id>', methods=['GET'])
def get_patient_alerts(patient_id):
    """Get alerts for a specific patient"""
    try:
        patient_alerts = [alert for alert in alerts_queue if alert['patient_id'] == patient_id]
        patient_alerts.reverse()  # Most recent first
        
        return jsonify({
            'patient_id': patient_id,
            'alerts': patient_alerts,
            'count': len(patient_alerts)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    try:
        global alerts_queue
        alerts_queue.clear()
        return jsonify({'message': 'All alerts cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/mark-read/<alert_id>', methods=['POST'])
def mark_alert_read(alert_id):
    """Mark a specific alert as read (placeholder for future implementation)"""
    try:
        # In a real implementation, you'd have alert IDs and mark them as read
        return jsonify({'message': f'Alert {alert_id} marked as read'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-all-status', methods=['POST'])
def reset_all_patient_status():
    """Reset all patient statuses (for testing)"""
    try:
        save_patient_status({})
        # Also clear alerts
        global alerts_queue
        alerts_queue.clear()
        return jsonify({'message': 'All patient statuses and alerts reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring a patient's ECG data"""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        ecg_data = data.get('ecg_data')
        
        if not patient_id or not ecg_data:
            return jsonify({'error': 'Missing patient_id or ecg_data'}), 400
            
        # Start monitoring this patient
        worker.start_monitoring(patient_id, ecg_data, handle_prediction)
        
        return jsonify({'message': f'Started monitoring patient {patient_id}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-monitoring/<patient_id>', methods=['POST'])
def stop_monitoring(patient_id):
    """Stop monitoring a specific patient"""
    try:
        worker.stop_monitoring(patient_id)
        return jsonify({'message': f'Stopped monitoring patient {patient_id}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-model-inference', methods=['POST'])
def start_model_inference():
    """Start model inference for all monitored patients"""
    try:
        success = worker.start_model_inference()
        if success:
            return jsonify({'message': 'Model inference started'})
        else:
            return jsonify({'error': 'Model not loaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-model-inference', methods=['POST'])
def stop_model_inference():
    """Stop model inference but keep monitoring active"""
    try:
        worker.stop_model_inference()
        return jsonify({'message': 'Model inference stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        model_info = {
            'model_loaded': worker.model is not None,
            'model_inference_active': worker.model_inference_active,
            'scaler_loaded': worker.scaler is not None,
            'model_name': worker.model.name if worker.model else None,
            'input_shape': str(worker.model.input_shape) if worker.model else None,
            'output_shape': str(worker.model.output_shape) if worker.model else None,
            'total_parameters': worker.model.count_params() if worker.model else None,
            'scaler_features': len(worker.scaler['mean']) if worker.scaler else None,
            'class_names': worker.logger.class_names
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the worker
    worker.start()
    # Clear any persisted patient statuses and alerts
    save_patient_status({})
    alerts_queue.clear()
    
    print("🚀 Starting ECG Monitoring Server...")
    print("📊 Server will be available at http://localhost:5000")
    print("📋 Available endpoints:")
    print("   - /api/monitoring/status - Get all patient monitoring status")
    print("   - /api/monitoring/patient/<id> - Get specific patient status")
    print("   - /api/monitoring/history/<id> - Get patient classification history")
    print("   - /api/alerts - Get all alerts")
    print("   - /api/start-monitoring - Start monitoring a patient")
    print("   - /api/stop-monitoring/<id> - Stop monitoring a patient")
    print("   - /api/model-info - Get model information")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        worker.stop() 
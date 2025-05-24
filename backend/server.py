from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Get the absolute path to the models directory
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'models'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
PATIENT_STATUS_FILE = os.path.join(DATA_DIR, 'patient_status.json')

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

@app.route('/api/reset-all-status', methods=['POST'])
def reset_all_patient_status():
    """Reset all patient statuses (for testing)"""
    try:
        save_patient_status({})
        return jsonify({'message': 'All patient statuses reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Serving model files from: {MODELS_DIR}")
    print(f"Serving data files from: {DATA_DIR}")
    print(f"Patient status file: {PATIENT_STATUS_FILE}")
    app.run(debug=True, port=5000) 
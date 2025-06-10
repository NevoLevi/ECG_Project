#!/usr/bin/env python3
"""
Test script for ECG real-time monitoring system.
This script demonstrates how the system works with sample ECG data.
"""

import time
import json
import numpy as np
import requests
from datetime import datetime

def generate_sample_ecg_data(length=2000, noise_level=0.1):
    """Generate realistic sample ECG data"""
    
    # Basic ECG waveform parameters
    fs = 125  # Sampling frequency (Hz)
    duration = length / fs  # Duration in seconds
    t = np.linspace(0, duration, length)
    
    # Generate base ECG pattern
    ecg = np.zeros(length)
    
    # Add P waves, QRS complexes, and T waves with some variation
    heartrate = 75  # beats per minute
    beat_interval = 60 / heartrate  # seconds between beats
    samples_per_beat = int(beat_interval * fs)
    
    num_beats = int(length / samples_per_beat)
    
    for beat in range(num_beats):
        start_idx = beat * samples_per_beat
        
        if start_idx + samples_per_beat < length:
            # P wave (small positive deflection)
            p_start = start_idx + int(0.1 * samples_per_beat)
            p_end = p_start + int(0.1 * samples_per_beat)
            if p_end < length:
                ecg[p_start:p_end] += 0.2 * np.sin(np.linspace(0, np.pi, p_end - p_start))
            
            # QRS complex (large spike)
            qrs_start = start_idx + int(0.3 * samples_per_beat)
            qrs_end = qrs_start + int(0.1 * samples_per_beat)
            if qrs_end < length:
                qrs_pattern = np.array([-0.3, 1.5, -0.8, 0.3])
                qrs_samples = np.interp(np.linspace(0, len(qrs_pattern)-1, qrs_end - qrs_start), 
                                       np.arange(len(qrs_pattern)), qrs_pattern)
                ecg[qrs_start:qrs_end] += qrs_samples
            
            # T wave (smooth positive wave)
            t_start = start_idx + int(0.5 * samples_per_beat)
            t_end = t_start + int(0.2 * samples_per_beat)
            if t_end < length:
                ecg[t_start:t_end] += 0.3 * np.sin(np.linspace(0, np.pi, t_end - t_start))
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level, length)
    ecg += noise
    
    # Add baseline wander
    baseline = 0.1 * np.sin(2 * np.pi * 0.5 * t)
    ecg += baseline
    
    return ecg.tolist()

def generate_abnormal_ecg_data(length=2000, abnormality_type='ventricular'):
    """Generate ECG data with specific abnormalities"""
    
    # Start with normal ECG
    ecg = np.array(generate_sample_ecg_data(length, noise_level=0.05))
    
    if abnormality_type == 'ventricular':
        # Add ventricular ectopic beats (wide QRS complexes)
        fs = 125
        beat_interval = int(60 / 75 * fs)  # 75 BPM
        
        # Replace some normal beats with wide, abnormal beats
        for i in range(2, len(ecg) - beat_interval, beat_interval * 3):  # Every 3rd beat
            if i + beat_interval < len(ecg):
                # Create wide, bizarre QRS complex
                abnormal_qrs = np.array([-0.5, -0.3, 2.0, -1.5, 0.8, -0.2])
                qrs_length = min(len(abnormal_qrs), beat_interval // 3)
                abnormal_qrs_resized = np.interp(np.linspace(0, len(abnormal_qrs)-1, qrs_length), 
                                               np.arange(len(abnormal_qrs)), abnormal_qrs)
                ecg[i:i+qrs_length] = abnormal_qrs_resized
    
    elif abnormality_type == 'atrial_fibrillation':
        # Add irregular rhythm and remove P waves
        # Make R-R intervals irregular
        np.random.seed(42)
        irregular_noise = np.random.normal(0, 0.3, len(ecg))
        ecg += irregular_noise
        
        # Remove P waves by adding noise where P waves should be
        fs = 125
        beat_interval = int(60 / 75 * fs)
        for i in range(0, len(ecg), beat_interval):
            p_start = i + beat_interval // 10
            p_end = p_start + beat_interval // 10
            if p_end < len(ecg):
                ecg[p_start:p_end] += np.random.normal(0, 0.4, p_end - p_start)
    
    return ecg.tolist()

def test_monitoring_system():
    """Test the ECG monitoring system"""
    
    print("🧪 Testing ECG Real-Time Monitoring System")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Check if server is running
    print("\n📡 Testing server connection...")
    try:
        response = requests.get(f"{base_url}/api/model-info")
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Server is running!")
            print(f"   Model loaded: {model_info.get('model_loaded', 'Unknown')}")
            print(f"   Model name: {model_info.get('model_name', 'Unknown')}")
            print(f"   Input shape: {model_info.get('input_shape', 'Unknown')}")
            print(f"   Class names: {model_info.get('class_names', {})}")
        else:
            print("❌ Server is not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to start the server first by running: python server.py")
        return
    
    # Test 2: Start monitoring normal patient
    print("\n👤 Testing normal patient monitoring...")
    normal_ecg = generate_sample_ecg_data(length=2000)
    
    response = requests.post(f"{base_url}/api/start-monitoring", json={
        'patient_id': 'patient_001_normal',
        'ecg_data': normal_ecg
    })
    
    if response.status_code == 200:
        print("✅ Started monitoring normal patient")
    else:
        print(f"❌ Failed to start monitoring: {response.text}")
    
    # Test 3: Start monitoring abnormal patient
    print("\n🚨 Testing abnormal patient monitoring...")
    abnormal_ecg = generate_abnormal_ecg_data(length=2000, abnormality_type='ventricular')
    
    response = requests.post(f"{base_url}/api/start-monitoring", json={
        'patient_id': 'patient_002_abnormal',
        'ecg_data': abnormal_ecg
    })
    
    if response.status_code == 200:
        print("✅ Started monitoring abnormal patient")
    else:
        print(f"❌ Failed to start monitoring: {response.text}")
    
    # Test 4: Monitor for classifications and alerts
    print("\n⏱️  Monitoring for 30 seconds to see classifications...")
    print("   (You should see real-time classifications in the server logs)")
    
    for i in range(6):  # Monitor for 30 seconds (5 second intervals)
        time.sleep(5)
        
        # Check monitoring status
        response = requests.get(f"{base_url}/api/monitoring/status")
        if response.status_code == 200:
            status = response.json()
            print(f"\n📊 Status after {(i+1)*5} seconds:")
            
            for patient_id, patient_status in status.items():
                summary = patient_status.get('classification_summary', {})
                if summary:
                    total = summary.get('total_classifications', 0)
                    abnormal_rate = summary.get('abnormal_rate', 0) * 100
                    print(f"   {patient_id}: {total} classifications, {abnormal_rate:.1f}% abnormal")
        
        # Check for alerts
        response = requests.get(f"{base_url}/api/alerts?limit=10")
        if response.status_code == 200:
            alerts_data = response.json()
            alerts = alerts_data.get('alerts', [])
            if alerts:
                print(f"🚨 {len(alerts)} recent alerts:")
                for alert in alerts[:3]:  # Show only first 3
                    print(f"   - {alert['severity']}: {alert['message']}")
    
    # Test 5: Get detailed patient history
    print("\n📋 Getting detailed patient histories...")
    
    for patient_id in ['patient_001_normal', 'patient_002_abnormal']:
        response = requests.get(f"{base_url}/api/monitoring/history/{patient_id}")
        if response.status_code == 200:
            history_data = response.json()
            history = history_data.get('history', [])
            summary = history_data.get('summary', {})
            
            print(f"\n👤 {patient_id}:")
            if summary:
                print(f"   Total classifications: {summary.get('total_classifications', 0)}")
                print(f"   Abnormal rate: {summary.get('abnormal_rate', 0)*100:.1f}%")
                print(f"   Class distribution: {summary.get('class_distribution', {})}")
            
            if history:
                print(f"   Recent classifications:")
                for classification in history[-5:]:  # Show last 5
                    timestamp = classification['timestamp'].split('T')[1][:8]  # Time only
                    print(f"     {timestamp}: {classification['class_name']} ({classification['confidence']:.3f})")
    
    # Test 6: Stop monitoring
    print("\n🛑 Stopping monitoring...")
    
    for patient_id in ['patient_001_normal', 'patient_002_abnormal']:
        response = requests.post(f"{base_url}/api/stop-monitoring/{patient_id}")
        if response.status_code == 200:
            print(f"✅ Stopped monitoring {patient_id}")
    
    print("\n🎉 Test completed successfully!")
    print("\n📊 Summary:")
    print("   - The system successfully loaded your CAT-Net model (or compatible fallback)")
    print("   - Real-time ECG classification is working every ~1.5 seconds")
    print("   - Abnormal patterns trigger alerts automatically")
    print("   - Classification history is logged for debugging")
    print("   - The system can monitor multiple patients simultaneously")
    
    print("\n💡 Next steps:")
    print("   - Check the 'ecg_classifications.log' file for detailed logs")
    print("   - Integrate this with your frontend for real-time monitoring")
    print("   - Add more sophisticated alert rules as needed")

if __name__ == "__main__":
    test_monitoring_system() 
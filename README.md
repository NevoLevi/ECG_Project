# ECG Hospital Monitoring System

A real-time ECG monitoring system that uses machine learning to analyze ECG signals and detect abnormalities in a hospital setting.

## Project Structure

```
ECG_Project/
├── backend/                    # Flask API server
│   └── server.py              # Main backend application
├── frontend/                   # React frontend application
│   ├── public/                # Static files
│   │   └── models/            # TensorFlow.js model files
│   ├── src/                   # React source code
│   ├── package.json           # Frontend dependencies
│   └── tailwind.config.js     # Tailwind CSS configuration
├── data/                      # Data files
│   ├── ecg_hospital_data_updated.csv  # Hospital ECG data (300 patients)
│   ├── mitbih_train.csv       # Original training data
│   ├── mitbih_test.csv        # Original test data
│   └── patient_status.json    # Patient status persistence
├── .venv/                     # Python virtual environment
├── generate_ecg_hospital_data_updated.py  # Data generation script
├── model_converter.py         # Model conversion utility
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **Real-time ECG Monitoring**: Monitors 300 patients simultaneously
- **Live ECG Visualization**: Moving waveforms with 125Hz sampling rate
- **ML-based Classification**: Background analysis every second using TensorFlow.js
- **Patient Status Management**: Persistent tracking of patient conditions
- **Doctor Workflow**: Review panel for medical staff with notes capability
- **Hospital Dashboard**: Professional interface showing patient counts and statuses

## Setup Instructions

### Backend Setup

1. Navigate to the project root directory
2. Activate the virtual environment:
   ```bash
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Flask server:
   ```bash
   cd backend
   python server.py
   ```
   The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the React development server:
   ```bash
   npm start
   ```
   The frontend will run on `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. The system will automatically load the hospital data and begin monitoring
3. Click on any patient in the list to view their live ECG
4. Use the doctor review panel to check patients and add notes
5. Monitor the dashboard for real-time patient status updates

## Data Format

Each patient has 18,700 ECG values (100 sequences × 187 values each), representing 149.6 seconds of continuous ECG data at 125Hz sampling rate.

## Patient Categories

- **290 Normal Patients**: Only contain normal ECG sequences (classification 0)
- **10 Mixed Patients**: Mostly normal with some abnormal sequences (classifications 1-3) randomly inserted within the first 50 sequences

## Technology Stack

- **Backend**: Flask, Python
- **Frontend**: React, TailwindCSS, Recharts
- **ML**: TensorFlow.js
- **Data**: CSV files with patient status JSON persistence 
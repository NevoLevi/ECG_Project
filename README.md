# ECG Hospital Monitoring System

A real-time ECG monitoring system that uses machine learning to analyze ECG signals and detect abnormalities in a hospital setting. This system provides live ECG visualization, AI-powered classification, and a comprehensive patient management interface for medical professionals.

![ECG Monitoring Dashboard](https://img.shields.io/badge/Status-Active-green)
![React](https://img.shields.io/badge/React-18.3.1-blue)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

### Core Functionality
- **Real-time ECG Monitoring**: Continuous monitoring of 47 patients with live ECG waveform display
- **AI-Powered Classification**: Automated detection of abnormal heartbeat patterns (V-type, S-type, F-type, Q-type)
- **Intelligent Alert System**: Smart patient status management with automatic prioritization
- **Doctor Review Interface**: Comprehensive patient review system with abnormal window visualization
- **Persistent Data Storage**: Patient status and classification history tracking

### Technical Features
- **Live ECG Visualization**: Smooth scrolling waveforms showing 1.496-second windows (187 samples at 125Hz)
- **Background AI Processing**: Non-intrusive classification every 7.48 seconds
- **Smart Status Updates**: Patients flagged as abnormal only when 2+ abnormal classifications detected in 5-window span
- **Dynamic Patient Sorting**: Abnormal patients automatically moved to top of list
- **Responsive UI**: Modern, medical-grade interface built with React and Tailwind CSS

## 📊 Project Structure

```
ECG_Project/
├── frontend/                           # React application
│   ├── public/
│   │   ├── cat_net_classification_report_fixed.csv
│   │   └── 47_Patients_Hospital.csv
│   ├── src/
│   │   ├── components/                 # React components
│   │   ├── services/                   # API and model services
│   │   └── App.js                      # Main application
│   ├── package.json
│   └── tailwind.config.js
├── static_model_test/                  # Model testing and data
│   ├── 47_Patients_Hospital.csv        # Patient ECG data (47 patients)
│   ├── cat_net_classification_report_20250609_042242.csv
│   ├── recreate_catnet.py              # Model recreation script
│   └── static_cat_net_test.py          # Model testing
├── backend/                            # Flask API (optional)
│   └── server.py
├── fix_classification_csv.py           # Data preprocessing utility
├── find_trigger_patients.py            # Patient analysis tool
├── verify_csv.py                       # Data verification script
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- Python 3.8+ (for data processing scripts)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/NevoLevi/ECG_Project.git
   cd ECG_Project
   ```

2. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

### Usage Instructions

1. **Load AI Model**: Click "🧠 Load AI Model" to initialize the system
2. **Load Hospital Data**: Click "📊 Load Hospital Data" and select the provided CSV file
3. **Start Monitoring**: Click "🤖 Start AI Monitoring" to begin real-time analysis
4. **Monitor Patients**: 
   - Select any patient to view their live ECG
   - Abnormal patients will automatically appear at the top with red highlighting
   - Click "Needs Check" buttons to review abnormal patterns
5. **Doctor Review**: View detailed abnormal ECG windows and mark patients as reviewed

## 📈 Data Format & Processing

### Patient Data
- **47 Patients** with complete ECG recordings
- **18,700 ECG samples per patient** (100 windows × 187 samples)
- **125Hz sampling rate** providing high-resolution cardiac monitoring
- **1.496-second display windows** for optimal visualization

### Classification System
- **Normal (N)**: Healthy heartbeat patterns
- **S-Type**: Supraventricular premature beat
- **V-Type**: Premature ventricular contraction  
- **F-Type**: Fusion of ventricular and normal beat
- **Q-Type**: Unclassifiable beat

### Smart Alert Logic
- Patients start with "Normal" status
- Status changes to "Needs Check" only when:
  - ≥2 abnormal classifications detected in any 5-window span
  - Majority abnormal class determines the patient's classification type
- Once flagged, patients remain in "Needs Check" until doctor review

## 🔧 Data Processing Scripts

### `fix_classification_csv.py`
Transforms the original classification CSV format:
- Reorders heartbeat sequences numerically (1, 2, 3... instead of 1, 10, 100...)
- Transposes data structure (patients as rows, heartbeats as columns)
- Ensures proper data alignment for real-time processing

### `find_trigger_patients.py`
Analyzes classification data to identify patients that should trigger abnormal status:
- Scans all 5-window groups for each patient
- Identifies patients with ≥2 abnormal classifications in same window group
- Provides detailed trigger analysis for system validation

### `verify_csv.py`
Data validation utility for ensuring correct CSV structure and content integrity.

## 🎯 Key Technical Achievements

### Real-time Performance
- **Smooth ECG Animation**: Continuous waveform display without interruptions during AI processing
- **Persistent Animation State**: ECG streams continue from correct positions across system updates
- **Optimized Re-rendering**: Smart component updates minimize performance impact

### AI Integration
- **Background Processing**: Non-blocking classification every 7.48 seconds
- **Smart Status Management**: Prevents false alarms with multi-window validation
- **Pattern Recognition**: Accurate identification of cardiac abnormalities

### User Experience
- **Medical-grade Interface**: Professional design suitable for clinical environments
- **Intelligent Prioritization**: Critical patients automatically highlighted
- **Comprehensive Review Tools**: Detailed abnormal pattern visualization for medical assessment

## 🏥 Clinical Workflow

1. **Continuous Monitoring**: System automatically monitors all patients
2. **Automatic Detection**: AI identifies abnormal patterns in background
3. **Smart Alerting**: Only patients with confirmed abnormalities are flagged
4. **Doctor Review**: Medical staff can examine specific abnormal windows
5. **Status Management**: Patients can be marked as reviewed and managed accordingly

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔬 Research Context

This system was developed as part of a Data Science in Industry project, demonstrating the practical application of machine learning in healthcare monitoring systems. The project showcases:

- Real-time data processing and visualization
- Machine learning model deployment in production-like environments  
- User interface design for critical healthcare applications
- Integration of multiple technologies for comprehensive system solutions

## 📞 Contact

For questions or support, please open an issue on this repository.

---

*Built with ❤️ for advancing healthcare technology through data science* 
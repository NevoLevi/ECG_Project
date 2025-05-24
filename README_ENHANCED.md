# Enhanced ECG Hospital Real-Time Monitor

An advanced real-time ECG monitoring system for hospitals with patient status management, live visualization, and AI-powered classification.

## 🚀 New Features & Enhancements

### ✅ **What's New in This Version:**

1. **New Hospital Data Format Support**
   - Supports the new `ecg_hospital_data_updated.csv` format
   - 300 patients with 18,700 ECG values each (100 sequences × 187 values)
   - Each patient represents 149.6 seconds of continuous ECG data

2. **Patient Status Management**
   - **Persistent status tracking**: Abnormal patients stay flagged until doctor checks
   - **Doctor check functionality**: Click to review and clear patients
   - **Status persistence**: Patient statuses saved across sessions
   - **Notes system**: Doctors can add notes during reviews

3. **Enhanced Real-Time Monitoring**
   - **Sliding window visualization**: 125 samples advance every second
   - **Live ECG display**: Real-time waveform with current classification
   - **Background classification**: Model runs every second on current window
   - **Smart alerts**: Only generates new alerts when status changes

4. **Improved User Interface**
   - **Patient dashboard**: Clear status overview (Normal/Needs Check/Under Review)
   - **Patient list**: Click to select and view live ECG
   - **Doctor panel**: Modal for patient review and status updates
   - **Alert management**: Recent alerts with confidence scores

## 📊 Hospital Data Structure

### Generated Patient Data:
- **Total Patients**: 300
- **Normal-only Patients**: 220 (sequences with classification 0 only)
- **Mixed Patients**: 80 (mostly normal with some abnormal in first 20 sequences)

### Data Format:
```
patient_id,ecg_0,ecg_1,ecg_2,...,ecg_18699
Patient_001,0.1627,0.5407,0.7558,...,0.3421
Patient_002,0.9601,0.8632,0.4615,...,0.2156
...
```

### Technical Specifications:
- **Sampling Rate**: 125 Hz
- **Window Size**: 187 samples (1.496 seconds)
- **Window Advancement**: 125 samples/second
- **Total Duration per Patient**: 149.6 seconds
- **Classification Classes**: 0=Normal, 1=S, 2=V, 3=F, 4=Q

## 🔧 Installation & Setup

### Prerequisites:
```bash
Python 3.8+
Node.js 14+
```

### Backend Setup:
```bash
# Install Python dependencies
pip install flask flask-cors pandas numpy requests

# Start backend server
python backend/server.py
```

### Frontend Setup:
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start frontend
npm start
```

### Model Setup:
Ensure your TensorFlow.js model files are in `frontend/public/models/`:
- `model.json` - Model architecture
- `group1-shard1of1.bin` - Model weights
- `scaler_params.json` - Preprocessing parameters

## 🎯 How to Use

### 1. **Load Hospital Data**
- Click "Load Hospital Data" to automatically load the 300 patients
- Or upload your own CSV file with the same format

### 2. **Start Monitoring**
- Load the AI model first
- Click "Start Monitoring" to begin real-time analysis
- System processes all 300 patients simultaneously

### 3. **Patient Management**
- **View Patient List**: See all patients with current status
- **Click Patient**: Select to view live ECG waveform
- **Status Indicators**:
  - 🟢 **Normal**: No abnormalities detected
  - 🔴 **Needs Check**: Abnormal pattern detected, requires doctor review
  - 🟡 **Under Review**: Doctor has reviewed but kept under observation

### 4. **Doctor Workflow**
- **Abnormal Detection**: System automatically flags patients with abnormal patterns
- **Review Process**: Click "Check" button on flagged patients
- **Doctor Actions**:
  - **Clear to Normal**: Reset patient status to normal
  - **Keep Under Review**: Acknowledge but maintain monitoring
  - **Add Notes**: Document findings and decisions

### 5. **Live ECG Visualization**
- Real-time waveform display for selected patient
- Updates every second with new 187-sample window
- Color-coded: Green for normal, Red for abnormal
- Shows current classification and confidence level

## 🔄 System Workflow

### Real-Time Processing:
1. **Every Second**:
   - Advance sliding window by 125 samples for each patient
   - Extract 187-sample window for AI classification
   - Update live ECG display for selected patient
   - Generate alerts for newly detected abnormalities

2. **Patient Status Updates**:
   - First abnormal detection → Status becomes "Needs Check"
   - Doctor review → Status becomes "Under Review" or "Normal"
   - Status persists until manually changed by doctor

3. **Background Operations**:
   - Continuous model inference on all patients
   - Status persistence to JSON file
   - Real-time UI updates

## 📊 API Endpoints

### Patient Status Management:
```
GET /api/patient-status - Get all patient statuses
GET /api/patient-status/{patient_id} - Get specific patient status
POST /api/patient-status/{patient_id} - Update patient status
POST /api/patient-status/{patient_id}/check - Doctor check patient
POST /api/reset-all-status - Reset all statuses (testing)
```

### Data Serving:
```
GET /data/ecg_hospital_data_updated.csv - Hospital ECG data
GET /models/model.json - AI model files
```

## 🧪 Testing

Run the verification script:
```bash
python test_app_functionality.py
```

This tests:
- Backend API endpoints
- Data format compatibility
- Patient status management
- Hospital data accessibility

## 📁 File Structure

```
ECG_Project/
├── data/
│   ├── ecg_hospital_data_updated.csv    # Generated hospital data
│   └── patient_status.json              # Patient status persistence
├── backend/
│   └── server.py                        # Enhanced Flask server
├── frontend/
│   ├── src/
│   │   ├── App.js                       # Enhanced React app
│   │   └── services/
│   │       └── modelService.js          # TensorFlow.js integration
│   │
└── public/
    └── models/                      # AI model files
```

## 🎮 Usage Example

1. **Start the System**:
   ```bash
   # Terminal 1: Backend
   python backend/server.py
   
   # Terminal 2: Frontend
   cd frontend && npm start
   ```

2. **Access the App**: Open http://localhost:3000

3. **Load Data**: Click "Load Hospital Data" (300 patients loaded)

4. **Load Model**: Click "Load AI Model"

5. **Start Monitoring**: Click "Start Monitoring"

6. **Interact**:
   - Click on patients to view their live ECG
   - Review flagged patients using doctor panel
   - Monitor real-time alerts and classifications

## 🔍 Key Improvements Over Previous Version

| Feature | Previous | Enhanced |
|---------|----------|----------|
| Data Format | Variable length CSV | Standardized 18,700 values per patient |
| Patient Status | Temporary predictions | Persistent status tracking |
| Doctor Workflow | None | Complete review and check-off system |
| UI | Basic monitoring | Professional hospital interface |
| Real-time Updates | Simple predictions | Live ECG with sliding window |
| Status Persistence | None | Backend storage with API |
| Scalability | Limited | Designed for 300+ patients |

## 🚨 Important Notes

- **Patient Status Persistence**: Patient statuses are saved to `data/patient_status.json`
- **Real-time Performance**: System processes 300 patients simultaneously every second
- **Doctor Workflow**: Critical for hospital use - abnormal patients stay flagged until reviewed
- **Data Format**: Specifically designed for the new hospital data structure
- **Browser Compatibility**: Tested with Chrome, Firefox, Edge

## 🎯 Perfect for Hospital Use

This enhanced system is now ready for real hospital deployment with:
- Professional patient status management
- Doctor workflow integration
- Persistent tracking across shifts
- Real-time monitoring of multiple patients
- Comprehensive alert system
- Proper data handling for 300+ patients

The system follows hospital protocols where abnormal patients remain flagged until a medical professional reviews and either clears or maintains their monitoring status. 
import React, { useState, useEffect, useCallback, useRef } from 'react';
import modelService from './services/modelService';
import './App.css';

// ECG Classification labels
const ECG_CLASSES = {
  0: { name: 'Normal', color: '#10B981', severity: 'low' },
  1: { name: 'S-Type Abnormal', color: '#F59E0B', severity: 'medium' },
  2: { name: 'V-Type Abnormal', color: '#6366F1', severity: 'medium' },
  3: { name: 'F-Type Abnormal', color: '#8B5CF6', severity: 'medium' },
  4: { name: 'Q-Type Abnormal', color: '#EF4444', severity: 'high' }
};

// Patient status types
const STATUS_TYPES = {
  NORMAL: { name: 'Normal', color: '#10B981', icon: '🟢' },
  NEEDS_CHECK: { name: 'Needs Check', color: '#EF4444', icon: '🔴' },
  UNDER_REVIEW: { name: 'Under Review', color: '#F59E0B', icon: '🟡' }
};

function App() {
  // State management
  const [modelLoaded, setModelLoaded] = useState(false);
  const [hospitalData, setHospitalData] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [currentPositions, setCurrentPositions] = useState({});
  const [patientPredictions, setPatientPredictions] = useState({});
  const [patientStatuses, setPatientStatuses] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [showDoctorModal, setShowDoctorModal] = useState(false);
  const [doctorNotes, setDoctorNotes] = useState('');
  const [patientToCheck, setPatientToCheck] = useState(null);
  const [showFileSelector, setShowFileSelector] = useState(false);

  const intervalRefs = useRef({});
  const positionRefs = useRef({});
  const fileInputRef = useRef(null);
  const intervalsRef = useRef({});

  // Add alert helper function
  const addAlert = useCallback((message, type = 'info') => {
    const alert = {
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      message,
      type
    };
    setAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep only last 10 alerts
  }, []);

  // Load model
  const handleLoadModel = async () => {
    try {
      setIsLoading(true);
      setLoadingStatus('Loading AI model...');
      
      await modelService.loadModel();
      setModelLoaded(true);
      addAlert('✅ AI Model loaded successfully', 'success');
    } catch (error) {
      addAlert(`❌ Error loading model: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  // Load hospital data
  const handleLoadHospitalData = async (event) => {
    if (event && event.target && event.target.files && event.target.files.length > 0) {
      // File was selected through the file input
      const file = event.target.files[0];
      
      try {
        setIsLoading(true);
        addAlert(`📊 Loading hospital data from ${file.name}...`, 'info');
        
        // Read the file
        const csvText = await readFileAsText(file);
        console.log(`Loaded CSV data with ${csvText.length} characters`);
        
        if (!csvText || csvText.trim().length === 0) {
          throw new Error('Loaded CSV file is empty');
        }
        
        // Parse CSV
        const parsedData = parseCSV(csvText);
        console.log(`Parsed ${parsedData.length} rows from CSV`);
        
        if (parsedData.length === 0) {
          throw new Error('No data rows found in CSV file');
        }
        
        setHospitalData(parsedData);
        
        // Extract patients from the data
        const extractedPatients = extractPatients(parsedData);
        console.log(`Extracted ${extractedPatients.length} patients with ECG data`);
        
        if (extractedPatients.length === 0) {
          throw new Error('No patients extracted from data. Check data format.');
        }
        
        setPatients(extractedPatients);
        setSelectedPatient(extractedPatients[0]);
        
        // Initialize positions for each patient
        const initialPositions = {};
        extractedPatients.forEach(patient => {
          initialPositions[patient.id] = 0;
        });
        setCurrentPositions(initialPositions);
        positionRefs.current = initialPositions;
        
        // Load patient statuses from the backend
        await fetchPatientStatuses();
        
        addAlert(`✅ Hospital data loaded successfully. Found ${extractedPatients.length} patients.`, 'success');
        setShowFileSelector(false);
        
      } catch (error) {
        console.error('Failed to load hospital data:', error);
        addAlert(`❌ Error: ${error.message}`, 'error');
      } finally {
        setIsLoading(false);
      }
    } else {
      // Show file selector
      setShowFileSelector(true);
    }
  };

  // Fetch patient statuses from backend
  const fetchPatientStatuses = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/patient-status');
      if (response.ok) {
        const statuses = await response.json();
        setPatientStatuses(statuses);
      } else {
        console.warn('Could not load patient statuses from backend');
      }
    } catch (error) {
      console.warn('Error fetching patient statuses:', error);
    }
  };

  // Update patient status
  const updatePatientStatus = async (patientId, status) => {
    try {
      setPatientStatuses(prev => ({
        ...prev,
        [patientId]: status
      }));
      
      // Update status on backend
      await fetch(`http://localhost:5000/api/patient-status/${patientId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
    } catch (error) {
      console.error('Failed to update patient status:', error);
    }
  };

  // Handle doctor check
  const handleDoctorCheck = (patient) => {
    setPatientToCheck(patient);
    setDoctorNotes('');
    setShowDoctorModal(true);
  };

  // Submit doctor check
  const submitDoctorCheck = async (action) => {
    try {
      const status = action === 'clear' ? 'NORMAL' : 'UNDER_REVIEW';
      
      await fetch(`http://localhost:5000/api/patient-status/${patientToCheck.id}/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status, notes: doctorNotes })
      });
      
      setPatientStatuses(prev => ({
        ...prev,
        [patientToCheck.id]: status
      }));
      
      addAlert(`✅ Patient ${patientToCheck.name} ${action === 'clear' ? 'cleared to normal' : 'kept under review'}`, 'success');
      
    } catch (error) {
      console.error('Failed to submit doctor check:', error);
      addAlert(`❌ Error: ${error.message}`, 'error');
    } finally {
      setShowDoctorModal(false);
      setPatientToCheck(null);
    }
  };

  // Parse CSV data
  const parseCSV = (csvText) => {
    // Split into lines and filter out empty lines
    let lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return [];

    // Remove header row (column names) before parsing patients
    lines.shift();
    console.log(`Found ${lines.length} data lines in CSV (header excluded)`);

    // Each row now represents a patient with ECG data
    return lines.map((line, index) => {
      const values = line.split(',');
      
      // Create a patient object with the ECG data
      const result = {
        patient_id: `Patient_${String(index + 1).padStart(3, '0')}`,
      };
      
      // Add each ECG value as a property
      values.forEach((value, i) => {
        if (!isNaN(parseFloat(value)) && value !== '') {
          result[`ecg_${i}`] = parseFloat(value);
        }
      });
      
      return result;
    });
  };

  // Extract patients from hospital data
  const extractPatients = (data) => {
    // This function processes data from the CSV file format
    
    const patients = [];
    
    // Each row in the CSV represents a single patient
    data.forEach((row, index) => {
      const patientId = row.patient_id || `Patient_${String(index + 1).padStart(3, '0')}`;
      
      // Extract ECG data from properties named ecg_*
      const ecgData = [];
      Object.entries(row).forEach(([key, value]) => {
        if (key.startsWith('ecg_') && !isNaN(parseFloat(value))) {
          ecgData.push(parseFloat(value));
        }
      });
      
      // If no ECG data was found, skip this patient
      if (ecgData.length === 0) {
        console.warn(`No ECG data found for patient ${patientId}`);
        return;
      }
      
      console.log(`Extracted ${ecgData.length} ECG points for patient ${patientId}`);
      
      // Generate patient metadata
      const age = Math.floor(Math.random() * 50) + 25; // Age between 25-74
      const gender = Math.random() > 0.5 ? 'Male' : 'Female';
      const roomNumber = `${String.fromCharCode(65 + Math.floor(index / 10))}-${(index % 10) + 1}${Math.floor(Math.random() * 10)}`;
      const heartRate = Math.floor(Math.random() * 40) + 60; // HR between 60-99
      const systolic = Math.floor(Math.random() * 40) + 110; // 110-149
      const diastolic = Math.floor(Math.random() * 30) + 60; // 60-89
      
      // Create patient object
      patients.push({
        id: patientId,
        name: `Patient ${patientId.split('_')[1] || index + 1}`,
        age,
        gender,
        roomNumber,
        heartRate,
        systolic,
        diastolic,
        ecgData,
        isAbnormal: index >= 280, // First 280 patients are normal, rest are abnormal per user instructions
      });
    });
    
    // If no patients with ECG data were found, log the issue
    if (patients.length === 0 && data.length > 0) {
      console.error('No patients with ECG data could be extracted. Data format may be incorrect.');
      console.log('First row sample:', data[0]);
    } else {
      console.log(`Successfully extracted ${patients.length} patients from data`);
    }
    
    return patients;
  };

  // Extract ECG window for display (375 samples = 3 seconds)
  const extractECGDisplayWindow = (patient, position) => {
    if (!patient || !patient.ecgData || patient.ecgData.length === 0) return [];
    
    const displayLength = 375; // 3 seconds * 125 Hz = 375 samples
    const startIdx = position;
    const endIdx = startIdx + displayLength;
    
    // Handle wrap-around if we reach the end of the data
    if (endIdx > patient.ecgData.length) {
      // Loop back to beginning if we reach the end
      const remaining = endIdx - patient.ecgData.length;
      console.log(`📊 Sliding window for ${patient.id}: wrapping around - taking ${patient.ecgData.length - startIdx} samples from end + ${remaining} from beginning`);
      return [
        ...patient.ecgData.slice(startIdx),
        ...patient.ecgData.slice(0, remaining)
      ];
    }
    
    // Regular case - no wrap around needed
    console.log(`📊 Sliding window for ${patient.id}: samples ${startIdx} to ${endIdx-1} (${displayLength} total)`);
    return patient.ecgData.slice(startIdx, endIdx);
  };

  // Start monitoring
  const handleStartMonitoring = async () => {
    if (!modelLoaded || !patients || !Array.isArray(patients) || patients.length === 0) {
      addAlert('❌ Please load model and hospital data first', 'error');
      return;
    }
    setIsMonitoring(true);
    addAlert('🚀 Starting patient monitoring...', 'info');
  };

  // Stop monitoring
  const handleStopMonitoring = async () => {
    setIsMonitoring(false);
    addAlert('🛑 Stopping patient monitoring...', 'info');
  };

  // ECG Wave Component with smooth sliding animation
  const ECGWave = React.memo(({ patient, monitoring, prediction }) => {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const gridCanvasRef = useRef(null);

    // Cache static grid background
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const { width, height } = canvas;
      const gridCanvas = document.createElement('canvas');
      gridCanvas.width = width;
      gridCanvas.height = height;
      const gctx = gridCanvas.getContext('2d');
      gctx.fillStyle = '#FFFFFF'; gctx.fillRect(0, 0, width, height);
      gctx.strokeStyle = '#F3F4F6'; gctx.lineWidth = 0.5;
      for (let x = 0; x < width; x += 25) { gctx.beginPath(); gctx.moveTo(x, 0); gctx.lineTo(x, height); gctx.stroke(); }
      for (let y = 0; y < height; y += 25) { gctx.beginPath(); gctx.moveTo(0, y); gctx.lineTo(width, y); gctx.stroke(); }
      gridCanvasRef.current = gridCanvas;
    }, []);

    useEffect(() => {
      if (!canvasRef.current || !patient || !patient.ecgData?.length) return;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const { width, height } = canvas;

      // Draw one sliding window frame
      const drawFrame = (pos) => {
        const ecgWindow = extractECGDisplayWindow(patient, pos);
        const maxPoints = 200;
        const factor = Math.max(1, Math.floor(ecgWindow.length / maxPoints));
        const windowData = ecgWindow.filter((_, i) => i % factor === 0);

        // Draw grid
        if (gridCanvasRef.current) ctx.drawImage(gridCanvasRef.current, 0, 0);
        else { ctx.fillStyle = '#FFFFFF'; ctx.fillRect(0, 0, width, height); }

        // Draw amplitude axis labels and midline
        ctx.save();
        ctx.fillStyle = '#4B5563'; // gray labels
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'left';
        // 1.0 at top
        ctx.textBaseline = 'top';
        ctx.fillText('1.0', 4, 4);
        // 0.5 in middle
        ctx.textBaseline = 'middle';
        ctx.fillText('0.5', 4, height / 2);
        // 0.0 at bottom
        ctx.textBaseline = 'bottom';
        ctx.fillText('0.0', 4, height - 4);
        ctx.restore();
        // Draw horizontal dashed midline
        ctx.save();
        ctx.strokeStyle = '#6B7280';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.restore();

        // Draw ECG path (map amplitude 0–1 to full height)
        const xStep = width / windowData.length;
        const grad = ctx.createLinearGradient(0, 0, width, 0);
        grad.addColorStop(0, '#93C5FD');
        grad.addColorStop(0.7, '#3B82F6');
        grad.addColorStop(1, '#1D4ED8');
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        windowData.forEach((v, i) => {
          const x = i * xStep;
          // v=1 at top (y=0), v=0 at bottom (y=height)
          const y = height * (1 - v);
          if (i === 0) ctx.moveTo(x, y);
          else {
            const px = (i - 1) * xStep;
            const py = height * (1 - windowData[i - 1]);
            ctx.quadraticCurveTo(px, py, (px + x) / 2, (py + y) / 2);
          }
        });
        ctx.stroke();

        // Draw red striped "current moment" indicator at 75% of canvas
        ctx.save();
        ctx.strokeStyle = '#EF4444';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        const markerX = width * 0.75;
        ctx.beginPath();
        ctx.moveTo(markerX, 0);
        ctx.lineTo(markerX, height);
        ctx.stroke();
        ctx.restore();
      };

      // Continuous scroll with wrap-around
      let startTime = null;
      const animate = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const pos = Math.floor((elapsed / 1000) * 125);
        const startIdx = pos % patient.ecgData.length;
        drawFrame(startIdx);
        if (monitoring) {
          animationRef.current = requestAnimationFrame(animate);
        }
      };

      if (monitoring) {
        startTime = null;
        animationRef.current = requestAnimationFrame(animate);
      } else {
        // static initial frame
        drawFrame(0);
      }

      return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current); };
    }, [patient, monitoring]);

    return (
      <canvas
        ref={canvasRef}
        width={600}
        height={200}
        className="border-2 border-gray-300 rounded-lg bg-white shadow-sm"
        style={{ display: 'block' }}
      />
    );
  });

  // Get status display for patient
  const getPatientStatusDisplay = (patientId) => {
    const status = patientStatuses[patientId] || 'NORMAL';
    return STATUS_TYPES[status] || STATUS_TYPES.NORMAL;
  };

  // Helper function to read file as text
  const readFileAsText = (file) => {
    return new Promise((resolve, reject) => {
      if (!file) {
        reject(new Error('No file provided'));
        return;
      }
      
      console.log(`Reading file: ${file.name}, size: ${file.size} bytes`);
      
      const reader = new FileReader();
      
      reader.onload = (event) => {
        const content = event.target.result;
        console.log(`File read successful, content length: ${content.length}`);
        resolve(content);
      };
      
      reader.onerror = (error) => {
        console.error('Error reading file:', error);
        reject(new Error('Failed to read file'));
      };
      
      reader.readAsText(file);
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-lg border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                          <span className="text-blue-600 mr-3">💓</span>
            ECG Hospital Monitoring System
          </h1>
          <p className="text-gray-600 mt-2">Real-time cardiac monitoring with AI-powered analysis</p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Control Panel */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Control Panel</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {/* Model Loading */}
            <button
              onClick={handleLoadModel}
              disabled={isLoading || modelLoaded}
              className={`p-4 rounded-lg font-medium transition-all ${
                modelLoaded
                  ? 'bg-green-100 text-green-800 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50'
              }`}
            >
              {modelLoaded ? '✅ Model Loaded' : '🤖 Load AI Model'}
            </button>

            {/* Hospital Data Loading */}
            <div className="relative">
              <button
                onClick={handleLoadHospitalData}
                disabled={isLoading || !modelLoaded}
                className={`p-4 rounded-lg font-medium transition-all w-full ${
                  hospitalData
                    ? 'bg-green-100 text-green-800'
                    : 'bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50'
                }`}
              >
                {hospitalData ? '✅ Data Loaded' : '📊 Load Hospital Data'}
              </button>
              
              {showFileSelector && (
                <div className="absolute top-full left-0 right-0 mt-2 p-4 bg-white shadow-lg rounded-lg z-10">
                  <h3 className="text-base font-medium mb-2">Select CSV file</h3>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleLoadHospitalData}
                    className="block w-full text-sm text-gray-500
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-full file:border-0
                      file:text-sm file:font-semibold
                      file:bg-purple-50 file:text-purple-700
                      hover:file:bg-purple-100"
                  />
                  <div className="mt-2 text-right">
                    <button 
                      onClick={() => setShowFileSelector(false)} 
                      className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Monitoring Control */}
            <button
              onClick={isMonitoring ? handleStopMonitoring : handleStartMonitoring}
              disabled={isLoading || !modelLoaded || !hospitalData}
              className={`p-4 rounded-lg font-medium transition-all ${
                isMonitoring
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-green-600 text-white hover:bg-green-700 disabled:opacity-50'
              }`}
            >
              {isMonitoring ? '⏹️ Stop Monitoring' : '▶️ Start Monitoring'}
            </button>
          </div>

          {/* Loading Status */}
          {isLoading && (
            <div className="flex items-center justify-center p-4 bg-blue-50 rounded-lg">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-3"></div>
              <span className="text-blue-800">{loadingStatus}</span>
            </div>
          )}

          {/* System Status */}
          <div className="flex flex-wrap gap-4 mt-4">
            <span className={`px-3 py-1 rounded-full text-sm ${
              modelLoaded ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
            }`}>
              Model: {modelLoaded ? 'Ready' : 'Not Loaded'}
            </span>
            <span className={`px-3 py-1 rounded-full text-sm ${
              hospitalData && patients.length > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
            }`}>
              Data: {hospitalData && patients.length > 0 ? `${patients.length} Patients` : 'Not Loaded'}
            </span>
            <span className={`px-3 py-1 rounded-full text-sm ${
              isMonitoring ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-600'
            }`}>
              Monitoring: {isMonitoring ? 'Active' : 'Inactive'}
            </span>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Patient List */}
          <div className="bg-white rounded-xl shadow-lg p-6 overflow-auto" style={{maxHeight: '800px'}}>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Patients ({patients.length})</h2>
            
            {patients.length > 0 ? (
              <div className="space-y-3">
                {patients.map(patient => {
                  const prediction = patientPredictions[patient.id];
                  const statusDisplay = getPatientStatusDisplay(patient.id);
                  
                  return (
                    <div 
                      key={patient.id} 
                      className={`border rounded-lg p-3 cursor-pointer transition-all ${
                        selectedPatient?.id === patient.id 
                          ? 'border-blue-500 bg-blue-50' 
                          : 'border-gray-200 hover:border-blue-300'
                      }`}
                      onClick={() => setSelectedPatient(patient)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="text-base font-medium text-gray-900 flex items-center">
                            <span className="mr-2">{statusDisplay.icon}</span>
                            {patient.name}
                          </h3>
                          <p className="text-sm text-gray-600">
                            Room: {patient.roomNumber} | Age: {patient.age}
                          </p>
                        </div>
                        
                        {prediction && (
                          <div 
                            className="px-2 py-1 rounded text-xs font-medium"
                            style={{ 
                              backgroundColor: `${prediction.classInfo?.color || '#3B82F6'}20`,
                              color: prediction.classInfo?.color || '#3B82F6' 
                            }}
                          >
                            {prediction.classInfo?.name || 'Unknown'}
                          </div>
                        )}
                      </div>
                      
                      {/* Actions */}
                      {patientStatuses[patient.id] === 'NEEDS_CHECK' && (
                        <div className="mt-2 flex justify-end">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDoctorCheck(patient);
                            }}
                            className="px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                          >
                            Check Patient
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <div className="text-6xl mb-4">👨‍⚕️</div>
                <p>Load hospital data to see patients</p>
              </div>
            )}
          </div>

          {/* Selected Patient Monitoring */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                {selectedPatient ? `Monitoring: ${selectedPatient.name}` : 'Patient Monitoring'}
              </h2>
              
              {selectedPatient ? (
                <div>
                  {/* Patient Info */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <h3 className="text-sm font-medium text-gray-500">Patient Info</h3>
                      <p className="text-base font-medium">
                        {selectedPatient.age} yr, {selectedPatient.gender}
                      </p>
                      <p className="text-sm text-gray-600">Room {selectedPatient.roomNumber}</p>
                    </div>
                    
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <h3 className="text-sm font-medium text-gray-500">Vital Signs</h3>
                      <p className="text-base font-medium">
                        HR: {selectedPatient.heartRate} bpm
                      </p>
                      <p className="text-sm text-gray-600">
                        BP: {selectedPatient.systolic}/{selectedPatient.diastolic} mmHg
                      </p>
                    </div>
                    
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <h3 className="text-sm font-medium text-gray-500">Status</h3>
                      <p className="text-base font-medium" style={{ color: getPatientStatusDisplay(selectedPatient.id).color }}>
                        {getPatientStatusDisplay(selectedPatient.id).name}
                      </p>
                      {patientPredictions[selectedPatient.id] && (
                        <p className="text-sm" style={{ color: patientPredictions[selectedPatient.id].classInfo?.color || '#3B82F6' }}>
                          {patientPredictions[selectedPatient.id].classInfo?.name || 'Unknown'} 
                          ({(patientPredictions[selectedPatient.id].confidence * 100).toFixed(1)}%)
                        </p>
                      )}
                    </div>
                  </div>
                  
                  {/* ECG Wave */}                  
                  <div className="bg-gray-50 p-4 rounded-lg">                    
                    <h3 className="text-sm font-medium text-gray-700 mb-2">
                      Live ECG (Sliding 3-second window)
                    </h3>
                    <div className="relative">
                      <ECGWave                         
                        patient={selectedPatient}                         
                        monitoring={isMonitoring}                         
                        prediction={patientPredictions[selectedPatient.id]}                      />                    
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-6xl mb-4">💓</div>
                  <p>Select a patient to monitor</p>
                </div>
              )}
            </div>
            
            {/* Alerts */}
            <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">System Alerts</h2>
              
              {alerts.length > 0 ? (
                <div className="space-y-3">
                  {alerts.map(alert => (
                    <div 
                      key={alert.id} 
                      className={`p-3 rounded-lg border ${
                        alert.type === 'error' 
                          ? 'bg-red-50 border-red-200 text-red-800' 
                          : alert.type === 'success'
                          ? 'bg-green-50 border-green-200 text-green-800'
                          : 'bg-blue-50 border-blue-200 text-blue-800'
                      }`}
                    >
                      <div className="flex items-start">
                        <div className="flex-grow">
                          <p className="text-sm">{alert.message}</p>
                          <p className="text-xs mt-1 opacity-70">{alert.timestamp}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  <div className="text-4xl mb-2">🔔</div>
                  <p>No alerts yet</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Doctor Check Modal */}
      {showDoctorModal && patientToCheck && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-lg w-full">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Doctor Check: {patientToCheck.name}
            </h2>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-1">Patient Status</h3>
                <div className="p-3 rounded-lg bg-red-50 text-red-800 border border-red-200">
                  Needs medical review
                </div>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-1">Doctor Notes</h3>
                <textarea
                  className="w-full p-3 border border-gray-300 rounded-lg"
                  rows={4}
                  placeholder="Add your observations and notes here..."
                  value={doctorNotes}
                  onChange={(e) => setDoctorNotes(e.target.value)}
                ></textarea>
              </div>
              
              <div className="flex justify-end space-x-3 pt-4">
                <button
                  onClick={() => setShowDoctorModal(false)}
                  className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={() => submitDoctorCheck('review')}
                  className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600"
                >
                  Keep Under Review
                </button>
                <button
                  onClick={() => submitDoctorCheck('clear')}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                >
                  Clear to Normal
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Footer */}
      <footer className="mt-auto py-6 text-center text-gray-500 text-sm">
        <p>ECG Hospital Monitoring System &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App; 
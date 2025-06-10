import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import modelService from './services/modelService';
import monitoringService from './services/monitoringService';
import './App.css';

// Classification full names mapping
const CLASSIFICATION_NAMES = {
  'S': 'Supraventricular Premature Beat',
  'V': 'Premature Ventricular Contraction',
  'F': 'Fusion of Ventricular and Normal Beat',
  'Q': 'Unclassifiable Beat',
  'N': 'Normal'
};

// These constants are kept for potential future use
// const ECG_CLASSES = {
//   0: { name: 'Normal', color: '#10B981', severity: 'low' },
//   1: { name: 'S-Type Abnormal', color: '#F59E0B', severity: 'medium' },
//   2: { name: 'V-Type Abnormal', color: '#6366F1', severity: 'medium' },
//   3: { name: 'F-Type Abnormal', color: '#8B5CF6', severity: 'medium' },
//   4: { name: 'Q-Type Abnormal', color: '#EF4444', severity: 'high' }
// };

// const STATUS_TYPES = {
//   NORMAL: { name: 'Normal', color: '#10B981', icon: '🟢' },
//   NEEDS_CHECK: { name: 'Needs Check', color: '#EF4444', icon: '🔴' },
//   UNDER_REVIEW: { name: 'Under Review', color: '#F59E0B', icon: '🟡' }
// };

// Component for displaying abnormal ECG windows
const AbnormalECGWindow = React.memo(({ patientId, windowIndex, classification, getECGWindowData }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    
    // Get ECG data for this specific window
    const ecgData = getECGWindowData(patientId, windowIndex);
    
    if (ecgData.length === 0) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Draw grid
    ctx.strokeStyle = '#E5E7EB';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let x = 0; x <= canvasWidth; x += canvasWidth / 6) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasHeight);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let y = 0; y <= canvasHeight; y += canvasHeight / 4) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
    }
    
    // Normalize ECG data
    const minVal = Math.min(...ecgData);
    const maxVal = Math.max(...ecgData);
    const range = maxVal - minVal;
    
    // Draw ECG wave
    const classificationColors = {
      'S': '#F59E0B', // Orange
      'V': '#6366F1', // Indigo  
      'F': '#8B5CF6', // Purple
      'Q': '#EF4444', // Red
      'N': '#10B981'  // Green
    };
    
    ctx.strokeStyle = classificationColors[classification] || '#3B82F6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const pixelsPerSample = canvasWidth / ecgData.length;
    
    ecgData.forEach((value, index) => {
      const x = index * pixelsPerSample;
      const normalizedValue = range > 0 ? (value - minVal) / range : 0.5;
      const y = canvasHeight - (normalizedValue * canvasHeight * 0.8 + canvasHeight * 0.1);
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Add classification label with full name
    ctx.fillStyle = classificationColors[classification] || '#3B82F6';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(CLASSIFICATION_NAMES[classification] || `${classification}-Type`, 10, 20);
    
  }, [patientId, windowIndex, classification, getECGWindowData]);
  
  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={150}
      className="w-full border border-gray-300 rounded bg-white"
    />
  );
});

function App() {
  // State management
  const [modelLoaded, setModelLoaded] = useState(false);
  const [hospitalData, setHospitalData] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [showFileSelector, setShowFileSelector] = useState(false);
  
  const [modelInfo, setModelInfo] = useState(null);
  
  // Patient status management for AI classification
  const [patientStatuses, setPatientStatuses] = useState({});
  const [classificationData, setClassificationData] = useState(null);
  const [classificationTimer, setClassificationTimer] = useState(null);
  const [currentWindowIndex, setCurrentWindowIndex] = useState(0);
  const [showAbnormalWindows, setShowAbnormalWindows] = useState(false);
  const [selectedAbnormalPatient, setSelectedAbnormalPatient] = useState(null);

  const ecgPositionsRef = useRef({});

  // Sort patients by status (needs check first) and ID, memoized
  const sortedPatients = useMemo(() => {
    // Create a shallow copy to avoid mutating state
    const list = patients.slice();
    list.sort((a, b) => {
      const aNum = parseInt(a.id.split('_')[1]);
      const bNum = parseInt(b.id.split('_')[1]);
      const aStatus = patientStatuses[aNum]?.status || 'NORMAL';
      const bStatus = patientStatuses[bNum]?.status || 'NORMAL';
      // Abnormal first
      if (aStatus === 'NEEDS_CHECK' && bStatus !== 'NEEDS_CHECK') return -1;
      if (bStatus === 'NEEDS_CHECK' && aStatus !== 'NEEDS_CHECK') return 1;
      // Then by numeric ID
      return aNum - bNum;
    });
    return list;
  }, [patients, patientStatuses]);

  const fileInputRef = useRef(null);
  
  // Cleanup classification timer on unmount
  useEffect(() => {
    return () => {
      if (classificationTimer) {
        clearInterval(classificationTimer);
      }
    };
  }, [classificationTimer]);
  
  // Add alert helper function
  const addAlert = useCallback((message, type = 'info') => {
    const alert = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toLocaleTimeString()
    };
    setAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep only last 10 alerts
  }, []);

  // Handle patient status actions
  const handlePatientAction = (patientId, action) => {
    setPatientStatuses(prev => {
      const newStatuses = { ...prev };
      const patient = newStatuses[patientId];
      
      if (!patient) return prev;
      
      switch (action) {
        case 'classify_normal':
          newStatuses[patientId] = {
            ...patient,
            status: 'NORMAL',
            classification: 'N',
            abnormalWindows: [],
            detectedAt: null,
            windowSpan: null
          };
          addAlert(`✅ Patient ${patientId.toString().padStart(3, '0')} classified as Normal`, 'success');
          break;
        case 'confirm':
          addAlert(`✅ Patient ${patientId.toString().padStart(3, '0')} abnormality confirmed`, 'warning');
          break;
        case 'mark_reviewed':
          newStatuses[patientId] = {
            ...patient,
            status: 'REVIEWED' // New status to indicate reviewed but still abnormal
          };
          addAlert(`✅ Patient ${patientId.toString().padStart(3, '0')} marked as reviewed`, 'success');
          break;
        default:
          break;
      }
      
      return newStatuses;
    });
  };

  // Load classification data from CSV
  const loadClassificationData = async () => {
    try {
      const response = await fetch('/cat_net_classification_report_fixed.csv');
      if (!response.ok) {
        throw new Error('Failed to load classification data');
      }
      const csvText = await response.text();
      
      // Parse classification CSV (new format: rows=patients, columns=heartbeats)
      const lines = csvText.split('\n').filter(line => line.trim() !== '');
      const classificationMap = {};
      
      lines.forEach((line, index) => {
        if (index === 0) return; // Skip header
        
        const values = line.split(',');
        const patientId = parseInt(values[0]); // First column is patient_id
        const classifications = values.slice(1); // All classification results (heartbeats 1-100)
        
        if (!isNaN(patientId)) {
          classificationMap[patientId] = classifications;
          console.log(`Loaded Patient ${patientId}: ${classifications.length} classifications`);
        }
      });
      
      setClassificationData(classificationMap);
      console.log('Classification data loaded:', Object.keys(classificationMap).length, 'patients');
      console.log('Sample data for patient 1:', classificationMap[1]?.slice(0, 10));
      addAlert(`✅ Classification data loaded successfully for ${Object.keys(classificationMap).length} patients`, 'success');
      return classificationMap;
    } catch (error) {
      addAlert(`❌ Error loading classification data: ${error.message}`, 'error');
      return null;
    }
  };

  // Load model and connect to backend
  const handleLoadModel = async () => {
    try {
      setIsLoading(true);
      setLoadingStatus('Connecting to monitoring server...');
      
      // Check backend connection and model status
      const modelInfo = await monitoringService.getModelInfo();
      // Clear any previous patient statuses and alerts on server
      await monitoringService.clearAllStatuses();
      // Clear client-side cached states
      setAlerts([]);
      setModelInfo(modelInfo);
      
      if (modelInfo.model_loaded) {
        setModelLoaded(true);
        addAlert(`✅ Connected to monitoring server - ${modelInfo.model_name || 'CAT-Net Compatible'} loaded`, 'success');
      } else {
        throw new Error('Backend model not loaded');
      }
      
    } catch (error) {
      console.error('Backend connection failed, trying frontend model...', error);
      
      // Fallback to frontend-only model
      try {
        setLoadingStatus('Loading frontend AI model...');
        await modelService.loadModel();
        setModelLoaded(true);
        addAlert('✅ Frontend AI Model loaded (limited functionality)', 'success');
      } catch (frontendError) {
        addAlert(`❌ Error: ${frontendError.message}`, 'error');
      }
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  // Process 5-window classification for a patient
  const processPatientClassification = (patientId, windowClassifications, windowStartIndex) => {
    if (!windowClassifications || windowClassifications.length !== 5) {
      console.log(`Invalid window classifications for Patient ${patientId}:`, windowClassifications);
      return;
    }
    
    // Only log to console, not to system alerts (too verbose)
    if (windowClassifications.some(cls => cls !== 'N' && cls !== '' && cls)) {
      console.log(`📊 Patient ${patientId.toString().padStart(3, '0')} Window ${windowStartIndex}-${windowStartIndex+4}: [${windowClassifications.join(', ')}]`);
    }
    
    // Count abnormal classifications
    const abnormalClasses = windowClassifications.filter(cls => cls !== 'N' && cls !== '' && cls);
    const abnormalCount = abnormalClasses.length;
    
    console.log(`Patient ${patientId}, Window ${windowStartIndex}-${windowStartIndex+4}:`, {
      classifications: windowClassifications,
      abnormalClasses,
      abnormalCount
    });
    
    // Check if patient needs status update (2+ abnormal in 5 windows)
    if (abnormalCount >= 2) {
      // Find majority abnormal class
      const classCounts = {};
      abnormalClasses.forEach(cls => {
        classCounts[cls] = (classCounts[cls] || 0) + 1;
      });
      
      const majorityClass = Object.entries(classCounts)
        .sort(([,a], [,b]) => b - a)[0][0];
      
      console.log(`Patient ${patientId} triggering abnormal status:`, {
        majorityClass,
        classCounts,
        windowClassifications
      });
      
      // Update patient status (avoid triggering re-renders of unrelated components)
      setPatientStatuses(prev => {
        // Only update if status actually changed
        const currentStatus = prev[patientId];
        if (currentStatus?.status === 'NEEDS_CHECK' && currentStatus?.classification === majorityClass) {
          return prev; // No change needed
        }
        
        return {
          ...prev,
          [patientId]: {
            status: 'NEEDS_CHECK',
            classification: majorityClass,
            abnormalWindows: windowClassifications.map((cls, idx) => ({
              windowIndex: windowStartIndex + idx,
              classification: cls,
              isAbnormal: cls !== 'N' && cls !== '' && cls
            })).filter(w => w.isAbnormal),
            detectedAt: new Date().toLocaleTimeString(),
            windowSpan: {
              start: windowStartIndex,
              end: windowStartIndex + 4
            }
          }
        };
      });
      
      addAlert(`🚨 Abnormal pattern detected in Patient ${patientId.toString().padStart(3, '0')}: ${majorityClass}-type (${abnormalCount}/5 abnormal)`, 'warning');
    }
  };

  // Start AI classification simulation
  const startClassification = () => {
    if (!classificationData || !patients.length) {
      addAlert('❌ Cannot start classification: missing data or patients', 'error');
      return;
    }
    
    // Initialize all patients as normal
    const initialStatuses = {};
    patients.forEach(patient => {
      const patientNum = parseInt(patient.id.split('_')[1]);
      initialStatuses[patientNum] = {
        status: 'NORMAL',
        classification: 'N',
        abnormalWindows: [],
        detectedAt: null,
        windowSpan: null
      };
    });
    setPatientStatuses(initialStatuses);
    
    // Start timer for classification every 5 windows (14.96 seconds - 2x slower)
    const intervalTime = (187 * 5 / 125) * 1000 * 2; // 14.96 seconds in milliseconds
    
    const timer = setInterval(() => {
      setCurrentWindowIndex(prev => {
        const newIndex = prev + 5;
        
        console.log(`🔄 Processing window ${newIndex} for classification updates`);
        
        // Process each patient
        patients.forEach(patient => {
          const patientNum = parseInt(patient.id.split('_')[1]);
          const patientClassifications = classificationData[patientNum];
          
          if (patientClassifications) {
            // Get 5 consecutive windows starting from newIndex
            if (newIndex + 4 < patientClassifications.length) {
              const windowClassifications = patientClassifications.slice(newIndex, newIndex + 5);
              processPatientClassification(patientNum, windowClassifications, newIndex);
            }
          }
        });
        
        return newIndex;
      });
    }, intervalTime);
    
    setClassificationTimer(timer);
    addAlert('🤖 AI classification started - monitoring every 14.96 seconds', 'info');
  };

  // Stop AI classification
  const stopClassification = () => {
    if (classificationTimer) {
      clearInterval(classificationTimer);
      setClassificationTimer(null);
      setCurrentWindowIndex(0);
      addAlert('⏹️ AI classification stopped', 'info');
    }
  };

  // Get ECG data for specific window
  const getECGWindowData = (patientId, windowIndex) => {
    const patient = patients.find(p => p.id === `Patient_${patientId.toString().padStart(3, '0')}`);
    if (!patient || !patient.ecgData) return [];
    
    const startIdx = windowIndex * 187;
    const endIdx = startIdx + 187;
    
    if (endIdx > patient.ecgData.length) return [];
    
    return patient.ecgData.slice(startIdx, endIdx);
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
        
        // Extract patients from the data
        const extractedPatients = extractPatients(parsedData);
        console.log(`Extracted ${extractedPatients.length} patients with ECG data`);
        
        if (extractedPatients.length === 0) {
          throw new Error('No patients extracted from data. Check data format.');
        }
        
        // Deduplicate patients by ID
        const uniquePatients = Array.from(
          new Map(extractedPatients.map(p => [p.id, p])).values()
        );
        setHospitalData(parsedData);
        setPatients(uniquePatients);
        setSelectedPatient(uniquePatients[0]);
        
        addAlert(`✅ Hospital data loaded successfully. Found ${extractedPatients.length} patients.`, 'success');
        setShowFileSelector(false);
        
        // Also load classification data
        await loadClassificationData();
        
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
    
    return patients;
  };

  // Extract ECG window for display (187 samples = 1.496 seconds)
  const extractECGDisplayWindow = (patient, position) => {
    if (!patient || !patient.ecgData || patient.ecgData.length === 0) return [];
    
    const displayLength = 187; // 1.496 seconds * 125 Hz = 187 samples
    const startIdx = position;
    const endIdx = startIdx + displayLength;
    
    // Handle wrap-around if we reach the end of the data
    if (endIdx > patient.ecgData.length) {
      // Loop back to beginning if we reach the end
      const remaining = endIdx - patient.ecgData.length;
      return [
        ...patient.ecgData.slice(startIdx),
        ...patient.ecgData.slice(0, remaining)
      ];
    }
    
    // Regular case - no wrap around needed
    return patient.ecgData.slice(startIdx, endIdx);
  };

  // ECG Wave Component with smooth sliding animation
  const ECGWave = React.memo(({ patient, isMonitoring, ecgPositionsRef }) => {
    const canvasRef = useRef(null);
    const animationFrameRef = useRef(null);
    const lastTimeRef = useRef(0);
    
    useEffect(() => {
      if (!patient || !patient.ecgData || patient.ecgData.length === 0) return;
      
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      // Retrieve the last known position for this patient, or default to 0
      let currentPosition = ecgPositionsRef.current[patient.id] || 0;
      
      const ctx = canvas.getContext('2d');
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      
      // Animation parameters
      const sampleRate = 125; // Hz
      const animationSpeed = 1 / 2; // 2x slower than real-time speed
      const samplesPerSecond = sampleRate * animationSpeed;
      const pixelsPerSample = canvasWidth / 187; // 187 samples = 1.496 seconds
      
      const drawFrame = (pos) => {
        // Clear canvas
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        
        // Draw grid
        ctx.strokeStyle = '#E5E7EB';
        ctx.lineWidth = 1;
        
        // Vertical grid lines (time markers)
        for (let x = 0; x <= canvasWidth; x += canvasWidth / 6) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, canvasHeight);
          ctx.stroke();
        }
        
        // Horizontal grid lines (amplitude markers)
        for (let y = 0; y <= canvasHeight; y += canvasHeight / 4) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvasWidth, y);
          ctx.stroke();
        }
        
        // Get ECG data for current window
        const ecgWindow = extractECGDisplayWindow(patient, Math.floor(pos));
        
        if (ecgWindow.length > 0) {
          // Normalize ECG data to canvas height
          const minVal = Math.min(...ecgWindow);
          const maxVal = Math.max(...ecgWindow);
          const range = maxVal - minVal;
          
          // Draw ECG wave
          ctx.strokeStyle = '#3B82F6';
          ctx.lineWidth = 2;
          ctx.beginPath();
          
          ecgWindow.forEach((value, index) => {
            const x = index * pixelsPerSample;
            const normalizedValue = range > 0 ? (value - minVal) / range : 0.5;
            const y = canvasHeight - (normalizedValue * canvasHeight * 0.8 + canvasHeight * 0.1);
            
            if (index === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          });
          
          ctx.stroke();
        }
      };
      
      const animate = (timestamp) => {
        const deltaTime = timestamp - lastTimeRef.current;
        lastTimeRef.current = timestamp;
        
        // Only update position if monitoring is active
        if (deltaTime > 0 && isMonitoring) {
          const samplesPerFrame = (samplesPerSecond * deltaTime) / 1000;
          currentPosition += samplesPerFrame;
          
          // Wrap around when we reach the end of the data
          if (currentPosition >= patient.ecgData.length) {
            currentPosition = 0;
          }

          // Persist the new position in the parent component's ref
          ecgPositionsRef.current[patient.id] = currentPosition;
        }
        
        drawFrame(currentPosition);
        animationFrameRef.current = requestAnimationFrame(animate);
      };
      
      lastTimeRef.current = performance.now();
      animationFrameRef.current = requestAnimationFrame(animate);
      
      // Cleanup
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    }, [patient, isMonitoring, ecgPositionsRef]);
    
    return (
      <canvas
        ref={canvasRef}
        width={800}
        height={200}
        className="w-full h-48 border border-gray-300 rounded bg-white"
        style={{ imageRendering: 'pixelated' }}
      />
    );
  });

  // Helper function to read file as text
  const readFileAsText = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => resolve(event.target.result);
      reader.onerror = (error) => reject(error);
      reader.readAsText(file);
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col">
      <div className="flex-1 p-6">
        {/* Header */}
        <div className="max-w-7xl mx-auto mb-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">
              🏥 ECG Hospital Monitoring System
            </h1>
            <p className="text-lg text-gray-600">
              Real-time ECG monitoring and analysis for hospital patients
            </p>
          </div>

          {/* Control Panel */}
          <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Control Panel</h2>
            
            <div className="space-y-4">
              {/* Model Controls */}
              <div className="space-y-3">
                <button
                  onClick={handleLoadModel}
                  disabled={isLoading}
                  className="w-full p-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 transition-all"
                >
                  🧠 Load AI Model
                </button>

                {/* Hospital Data Controls */}
                <div className="relative">
                  <button
                    onClick={handleLoadHospitalData}
                    disabled={isLoading}
                    className="w-full p-4 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:opacity-50 transition-all"
                  >
                    📊 Load Hospital Data
                  </button>
                  
                  {showFileSelector && (
                    <div className="absolute top-full left-0 right-0 mt-2 p-4 bg-white border border-gray-300 rounded-lg shadow-lg z-10">
                      <p className="text-sm text-gray-600 mb-2">Select a CSV file:</p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv"
                        onChange={handleLoadHospitalData}
                        className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
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

                {/* AI Classification Controls */}
                {hospitalData && patients.length > 0 && classificationData && (
                  <div className="space-y-3">
                    <div className="flex space-x-2">
                      <button
                        onClick={startClassification}
                        disabled={!!classificationTimer}
                        className="flex-1 p-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 transition-all"
                      >
                        🤖 Start AI Monitoring
                      </button>
                      <button
                        onClick={stopClassification}
                        disabled={!classificationTimer}
                        className="flex-1 p-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 disabled:opacity-50 transition-all"
                      >
                        ⏹️ Stop AI Monitoring
                      </button>
                    </div>
                    
                    {classificationTimer && (
                      <div className="text-center p-2 bg-purple-50 rounded-lg">
                        <p className="text-sm text-purple-700">
                          🔄 AI Monitoring Active - Window {currentWindowIndex}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* System Status */}
            <div className="flex flex-wrap gap-4 mt-4">
              <span className={`px-3 py-1 rounded-full text-sm ${
                modelLoaded ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
              }`}>
                Model: {modelLoaded ? (modelInfo?.model_name || 'Ready') : 'Not Loaded'}
              </span>
              <span className={`px-3 py-1 rounded-full text-sm ${
                hospitalData && patients.length > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
              }`}>
                Data: {hospitalData && patients.length > 0 ? `${patients.length} Patients` : 'Not Loaded'}
              </span>
              <span className={`px-3 py-1 rounded-full text-sm ${
                classificationData ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
              }`}>
                AI Data: {classificationData ? 'Ready' : 'Not Loaded'}
              </span>
            </div>
          </div>

          {/* Loading Status */}
          {isLoading && (
            <div className="flex items-center justify-center p-4 bg-blue-50 rounded-lg">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-3"></div>
              <span className="text-blue-800">{loadingStatus}</span>
            </div>
          )}
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Patient List */}
          <div className="bg-white rounded-xl shadow-lg p-6 overflow-auto" style={{maxHeight: '800px'}}>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Patients ({sortedPatients.length})</h2>
            
            {sortedPatients.length > 0 ? (
              <div className="space-y-3">
                {sortedPatients.map(patient => {
                  const patientNum = parseInt(patient.id.split('_')[1]);
                  const status = patientStatuses[patientNum] || { status: 'NORMAL', classification: 'N' };
                  
                  return (
                    <div
                      key={patient.id}
                      onClick={() => setSelectedPatient(patient)}
                      className={`p-4 rounded-lg cursor-pointer transition-all border-2 ${
                        selectedPatient?.id === patient.id
                          ? 'border-blue-500 bg-blue-50 shadow-md'
                          : status.status === 'NEEDS_CHECK'
                          ? 'border-red-200 bg-red-50 hover:bg-red-100'
                          : 'border-gray-200 bg-white hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2">
                            <div className="font-semibold text-gray-900">
                              {patient.name}
                            </div>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              status.status === 'NEEDS_CHECK' 
                                ? 'bg-red-100 text-red-800' 
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {status.status === 'NEEDS_CHECK' ? `${status.classification}-Type` : 'Normal'}
                            </span>
                          </div>
                          <div className="text-sm text-gray-600">
                            ID: {patient.id}
                          </div>
                          {(status.status === 'NEEDS_CHECK' || status.status === 'REVIEWED') && (
                            <div className="text-xs text-red-600 mt-1">
                              Detected at {status.detectedAt}
                            </div>
                          )}
                        </div>
                        {(status.status === 'NEEDS_CHECK' || status.status === 'REVIEWED') && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedAbnormalPatient(patientNum);
                              setShowAbnormalWindows(true);
                            }}
                            className={`px-3 py-1 text-white text-xs rounded-full transition-colors ${
                              status.status === 'REVIEWED' 
                                ? 'bg-orange-600 hover:bg-orange-700' 
                                : 'bg-red-600 hover:bg-red-700'
                            }`}
                          >
                            {status.status === 'REVIEWED' ? 'Reviewed' : 'Needs Check'}
                          </button>
                        )}
                      </div>
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
                      <h3 className="text-sm font-medium text-gray-500">Patient</h3>
                      <div className="flex items-center justify-between">
                        <div>
                          <h2 className="text-xl font-semibold text-gray-900">{selectedPatient.name}</h2>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* ECG Wave */}                  
                  <div className="bg-gray-50 p-4 rounded-lg">                    
                    <h3 className="text-sm font-medium text-gray-700 mb-2">
                      Live ECG (Sliding 1.496-second window)
                    </h3>
                    <div className="relative">
                      <ECGWave 
                        patient={selectedPatient} 
                        isMonitoring={!!classificationTimer}
                        ecgPositionsRef={ecgPositionsRef}
                      />                    
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
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-gray-800">System Alerts</h2>
                {alerts.length > 0 && (
                  <button
                    onClick={() => setAlerts([])}
                    className="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
                  >
                    Clear All
                  </button>
                )}
              </div>
              
              <div className="h-60 overflow-y-auto border border-gray-200 rounded-lg">
                {alerts.length > 0 ? (
                  <div className="p-3 space-y-3">
                    {alerts.map(alert => (
                      <div 
                        key={alert.id} 
                        className={`p-3 rounded-lg border ${
                          alert.type === 'error' 
                            ? 'bg-red-50 border-red-200 text-red-800' 
                            : alert.type === 'success'
                            ? 'bg-green-50 border-green-200 text-green-800'
                            : alert.type === 'warning'
                            ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
                            : 'bg-blue-50 border-blue-200 text-blue-800'
                        }`}
                      >
                        <div className="flex items-start">
                          <div className="flex-grow">
                            <p className="text-sm font-medium">{alert.message}</p>
                            <p className="text-xs mt-1 opacity-70">{alert.timestamp}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <div className="text-4xl mb-2">🔔</div>
                      <p className="text-sm">No alerts yet</p>
                      <p className="text-xs mt-1">System messages will appear here</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Abnormal Windows Modal */}
      {showAbnormalWindows && selectedAbnormalPatient && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900">
                  Abnormal ECG Windows - Patient {selectedAbnormalPatient.toString().padStart(3, '0')}
                </h2>
                <button
                  onClick={() => {
                    setShowAbnormalWindows(false);
                    setSelectedAbnormalPatient(null);
                  }}
                  className="text-gray-500 hover:text-gray-700 text-2xl"
                >
                  ×
                </button>
              </div>
              
              {patientStatuses[selectedAbnormalPatient]?.abnormalWindows && (
                <div className="space-y-6">
                  <div className="bg-red-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-red-800 mb-2">Detection Summary</h3>
                    <p className="text-sm text-red-700">
                      Classification: <strong>{CLASSIFICATION_NAMES[patientStatuses[selectedAbnormalPatient].classification] || `${patientStatuses[selectedAbnormalPatient].classification}-Type Abnormal`}</strong>
                    </p>
                    <p className="text-sm text-red-700">
                      Detected at: {patientStatuses[selectedAbnormalPatient].detectedAt}
                    </p>
                    <p className="text-sm text-red-700">
                      Window span: {patientStatuses[selectedAbnormalPatient].windowSpan.start} - {patientStatuses[selectedAbnormalPatient].windowSpan.end}
                    </p>
                  </div>
                  
                  {patientStatuses[selectedAbnormalPatient].abnormalWindows.map((window, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">
                        Window {window.windowIndex} - {CLASSIFICATION_NAMES[window.classification] || `${window.classification}-Type Classification`}
                      </h4>
                      <AbnormalECGWindow 
                        patientId={selectedAbnormalPatient}
                        windowIndex={window.windowIndex}
                        classification={window.classification}
                        getECGWindowData={getECGWindowData}
                      />
                    </div>
                  ))}
                </div>
              )}
              
              <div className="mt-6 flex justify-end space-x-3">
                <button
                  onClick={() => {
                    setShowAbnormalWindows(false);
                    setSelectedAbnormalPatient(null);
                  }}
                  className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Close
                </button>
                <button
                  onClick={() => {
                    handlePatientAction(selectedAbnormalPatient, 'classify_normal');
                    setShowAbnormalWindows(false);
                    setSelectedAbnormalPatient(null);
                  }}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  Classify as Normal
                </button>
                <button
                  onClick={() => {
                    handlePatientAction(selectedAbnormalPatient, 'confirm');
                    setShowAbnormalWindows(false);
                    setSelectedAbnormalPatient(null);
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Confirm
                </button>
                <button
                  onClick={() => {
                    handlePatientAction(selectedAbnormalPatient, 'mark_reviewed');
                    setShowAbnormalWindows(false);
                    setSelectedAbnormalPatient(null);
                  }}
                  className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
                >
                  Mark as Reviewed
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
/**
 * Service for real-time ECG monitoring backend communication
 */

const BASE_URL = 'http://localhost:5000';

class MonitoringService {
  constructor() {
    this.alertCallbacks = [];
    this.statusCallbacks = [];
    this.pollingInterval = null;
  }

  // Model and system info
  async getModelInfo() {
    try {
      const response = await fetch(`${BASE_URL}/api/model-info`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error getting model info:', error);
      throw new Error('Failed to connect to monitoring server');
    }
  }

  // Patient monitoring
  async startMonitoring(patientId, ecgData) {
    try {
      const response = await fetch(`${BASE_URL}/api/start-monitoring`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          ecg_data: ecgData
        })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error starting monitoring:', error);
      throw error;
    }
  }

  async stopMonitoring(patientId) {
    try {
      const response = await fetch(`${BASE_URL}/api/stop-monitoring/${patientId}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error stopping monitoring:', error);
      throw error;
    }
  }

  // Model inference control
  async startModelInference() {
    try {
      const response = await fetch(`${BASE_URL}/api/start-model-inference`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error starting model inference:', error);
      throw error;
    }
  }

  async stopModelInference() {
    try {
      const response = await fetch(`${BASE_URL}/api/stop-model-inference`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error stopping model inference:', error);
      throw error;
    }
  }

  // Status and history
  async getMonitoringStatus() {
    try {
      const response = await fetch(`${BASE_URL}/api/monitoring/status`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error getting monitoring status:', error);
      return {};
    }
  }

  async getPatientStatus(patientId) {
    try {
      const response = await fetch(`${BASE_URL}/api/monitoring/patient/${patientId}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error getting patient status:', error);
      return null;
    }
  }

  async getPatientHistory(patientId, limit = 50) {
    try {
      const response = await fetch(`${BASE_URL}/api/monitoring/history/${patientId}?limit=${limit}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error getting patient history:', error);
      return { history: [], summary: null };
    }
  }

  // Alerts
  async getAlerts(limit = 50) {
    try {
      const response = await fetch(`${BASE_URL}/api/alerts?limit=${limit}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error getting alerts:', error);
      return { alerts: [], total_count: 0 };
    }
  }

  async getPatientAlerts(patientId) {
    try {
      const response = await fetch(`${BASE_URL}/api/alerts/${patientId}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error getting patient alerts:', error);
      return { alerts: [], count: 0 };
    }
  }

  async clearAlerts() {
    try {
      const response = await fetch(`${BASE_URL}/api/alerts/clear`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error clearing alerts:', error);
      throw error;
    }
  }

  // Real-time polling
  startPolling(interval = 3000) {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
    }

    this.pollingInterval = setInterval(async () => {
      try {
        // Poll for monitoring status
        const status = await this.getMonitoringStatus();
        this.statusCallbacks.forEach(callback => {
          try {
            callback(status);
          } catch (error) {
            console.error('Error in status callback:', error);
          }
        });

        // Poll for new alerts
        const alertsData = await this.getAlerts(10);
        this.alertCallbacks.forEach(callback => {
          try {
            callback(alertsData.alerts);
          } catch (error) {
            console.error('Error in alert callback:', error);
          }
        });

      } catch (error) {
        console.error('Error during polling:', error);
      }
    }, interval);
  }

  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  }

  // Callback management
  onStatusUpdate(callback) {
    this.statusCallbacks.push(callback);
    return () => {
      const index = this.statusCallbacks.indexOf(callback);
      if (index > -1) {
        this.statusCallbacks.splice(index, 1);
      }
    };
  }

  onNewAlert(callback) {
    this.alertCallbacks.push(callback);
    return () => {
      const index = this.alertCallbacks.indexOf(callback);
      if (index > -1) {
        this.alertCallbacks.splice(index, 1);
      }
    };
  }

  // Legacy patient status (keeping for compatibility)
  async updatePatientStatus(patientId, status) {
    try {
      const response = await fetch(`${BASE_URL}/api/patient-status/${patientId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error updating patient status:', error);
      throw error;
    }
  }

  async checkPatient(patientId, notes, action = 'review') {
    try {
      const response = await fetch(`${BASE_URL}/api/patient-status/${patientId}/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          notes, 
          reset_to_normal: action === 'clear' 
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error checking patient:', error);
      throw error;
    }
  }

  async clearAllStatuses() {
    try {
      const response = await fetch(`${BASE_URL}/api/reset-all-status`, {
        method: 'POST'
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error clearing all statuses:', error);
      throw error;
    }
  }
}

// Create and export singleton instance
const monitoringService = new MonitoringService();
export default monitoringService; 
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import modelService from './services/modelService';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

window.modelService = modelService; 
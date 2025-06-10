import React, { useEffect, useRef, useState } from 'react';
import { useTheme } from '../contexts/ThemeContext';

const BUFFER_SIZE = 1000; // Number of points to keep in buffer
const RENDER_INTERVAL = 16; // ~60 FPS

const ECGVisualizer = ({ data, width = 800, height = 200 }) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const dataBufferRef = useRef([]);
  const lastRenderTimeRef = useRef(0);
  const { theme } = useTheme();
  
  // Initialize canvas and context
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = width;
    canvas.height = height;
    
    // Set initial styles
    ctx.strokeStyle = theme.colors.primary;
    ctx.lineWidth = 2;
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height, theme]);

  // Update data buffer
  useEffect(() => {
    if (data) {
      dataBufferRef.current = [...dataBufferRef.current, ...data].slice(-BUFFER_SIZE);
    }
  }, [data]);

  // Render loop
  useEffect(() => {
    const render = (timestamp) => {
      if (timestamp - lastRenderTimeRef.current >= RENDER_INTERVAL) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        drawGrid(ctx, width, height, theme);
        
        // Draw ECG line
        drawECGLine(ctx, dataBufferRef.current, width, height, theme);
        
        lastRenderTimeRef.current = timestamp;
      }
      
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    animationFrameRef.current = requestAnimationFrame(render);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height, theme]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ background: theme.colors.background }}
      />
    </div>
  );
};

function drawGrid(ctx, width, height, theme) {
  const gridSize = 20;
  ctx.strokeStyle = theme.colors.grid;
  ctx.lineWidth = 0.5;
  
  // Draw vertical lines
  for (let x = 0; x <= width; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  
  // Draw horizontal lines
  for (let y = 0; y <= height; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawECGLine(ctx, data, width, height, theme) {
  if (!data.length) return;
  
  const xStep = width / (data.length - 1);
  const yScale = height / 2;
  const yOffset = height / 2;
  
  ctx.beginPath();
  ctx.strokeStyle = theme.colors.primary;
  ctx.lineWidth = 2;
  
  data.forEach((value, index) => {
    const x = index * xStep;
    const y = yOffset - (value * yScale);
    
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  
  ctx.stroke();
}

export default ECGVisualizer; 
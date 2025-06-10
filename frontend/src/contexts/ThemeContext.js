import React, { createContext, useContext, useState } from 'react';

const themes = {
  light: {
    colors: {
      primary: '#2563eb',
      secondary: '#4b5563',
      background: '#ffffff',
      text: '#1f2937',
      grid: '#e5e7eb',
      success: '#10b981',
      warning: '#f59e0b',
      error: '#ef4444'
    }
  },
  dark: {
    colors: {
      primary: '#3b82f6',
      secondary: '#9ca3af',
      background: '#1f2937',
      text: '#f3f4f6',
      grid: '#374151',
      success: '#34d399',
      warning: '#fbbf24',
      error: '#f87171'
    }
  }
};

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(themes.light);

  const toggleTheme = () => {
    setTheme(prevTheme => 
      prevTheme === themes.light ? themes.dark : themes.light
    );
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
} 
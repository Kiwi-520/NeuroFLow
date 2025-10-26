// Calming, soothing theme system for neurodivergent users
export interface ThemeColors {
  // Core brand colors - soft and calming
  primary: {
    50: string;
    100: string;
    200: string;
    300: string;
    400: string;
    500: string;
    600: string;
    700: string;
    800: string;
    900: string;
  };
  
  // Gentle secondary colors
  secondary: {
    50: string;
    100: string;
    200: string;
    300: string;
    400: string;
    500: string;
    600: string;
    700: string;
    800: string;
    900: string;
  };
  
  // Soft background colors
  background: {
    primary: string;
    secondary: string;
    tertiary: string;
    paper: string;
  };
  
  // Gentle text colors
  text: {
    primary: string;
    secondary: string;
    disabled: string;
    hint: string;
  };
  
  // Emotion-based colors
  emotion: {
    calm: string;
    focused: string;
    energized: string;
    relaxed: string;
    positive: string;
    warning: string;
    error: string;
    success: string;
  };
  
  // Accessibility colors
  accessibility: {
    highContrast: boolean;
    focusRing: string;
    selectedBg: string;
    hoverBg: string;
  };
}

export interface ThemeConfig {
  mode: 'light' | 'dark' | 'auto';
  reducedMotion: boolean;
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large';
  cornerRadius: 'none' | 'small' | 'medium' | 'large';
  colorBlindFriendly: boolean;
  fontFamily: 'inter' | 'poppins' | 'system';
}

// Soothing light theme
export const lightTheme: ThemeColors = {
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
  },
  
  secondary: {
    50: '#fdf4ff',
    100: '#fae8ff',
    200: '#f5d0fe',
    300: '#f0abfc',
    400: '#e879f9',
    500: '#d946ef',
    600: '#c026d3',
    700: '#a21caf',
    800: '#86198f',
    900: '#701a75',
  },
  
  background: {
    primary: '#fefefe',
    secondary: '#f8fafc',
    tertiary: '#f1f5f9',
    paper: '#ffffff',
  },
  
  text: {
    primary: '#1e293b',
    secondary: '#64748b',
    disabled: '#94a3b8',
    hint: '#cbd5e1',
  },
  
  emotion: {
    calm: '#bfdbfe',     // Soft blue
    focused: '#c7d2fe',  // Gentle purple
    energized: '#fed7aa', // Warm orange
    relaxed: '#bbf7d0',  // Soft green
    positive: '#fef3c7', // Gentle yellow
    warning: '#fed7aa',  // Soft orange
    error: '#fecaca',    // Gentle red
    success: '#bbf7d0',  // Soft green
  },
  
  accessibility: {
    highContrast: false,
    focusRing: '#3b82f6',
    selectedBg: '#dbeafe',
    hoverBg: '#f1f5f9',
  },
};

// Calming dark theme
export const darkTheme: ThemeColors = {
  primary: {
    50: '#0c1821',
    100: '#1e293b',
    200: '#334155',
    300: '#475569',
    400: '#64748b',
    500: '#94a3b8',
    600: '#cbd5e1',
    700: '#e2e8f0',
    800: '#f1f5f9',
    900: '#f8fafc',
  },
  
  secondary: {
    50: '#1a0d1f',
    100: '#2d1b3d',
    200: '#44285f',
    300: '#5b3580',
    400: '#7c3aed',
    500: '#8b5cf6',
    600: '#a78bfa',
    700: '#c4b5fd',
    800: '#ddd6fe',
    900: '#ede9fe',
  },
  
  background: {
    primary: '#0f172a',
    secondary: '#1e293b',
    tertiary: '#334155',
    paper: '#1e293b',
  },
  
  text: {
    primary: '#f8fafc',
    secondary: '#cbd5e1',
    disabled: '#64748b',
    hint: '#475569',
  },
  
  emotion: {
    calm: '#1e3a8a',     // Deep blue
    focused: '#4c1d95',  // Deep purple
    energized: '#9a3412', // Warm brown
    relaxed: '#166534',  // Deep green
    positive: '#a16207', // Deep yellow
    warning: '#ea580c',  // Orange
    error: '#dc2626',    // Red
    success: '#16a34a',  // Green
  },
  
  accessibility: {
    highContrast: false,
    focusRing: '#60a5fa',
    selectedBg: '#1e3a8a',
    hoverBg: '#334155',
  },
};

// High contrast themes for accessibility
export const highContrastLight: ThemeColors = {
  ...lightTheme,
  background: {
    primary: '#ffffff',
    secondary: '#ffffff',
    tertiary: '#f5f5f5',
    paper: '#ffffff',
  },
  text: {
    primary: '#000000',
    secondary: '#333333',
    disabled: '#666666',
    hint: '#999999',
  },
  accessibility: {
    highContrast: true,
    focusRing: '#000000',
    selectedBg: '#000000',
    hoverBg: '#f0f0f0',
  },
};

export const highContrastDark: ThemeColors = {
  ...darkTheme,
  background: {
    primary: '#000000',
    secondary: '#000000',
    tertiary: '#1a1a1a',
    paper: '#000000',
  },
  text: {
    primary: '#ffffff',
    secondary: '#cccccc',
    disabled: '#999999',
    hint: '#666666',
  },
  accessibility: {
    highContrast: true,
    focusRing: '#ffffff',
    selectedBg: '#ffffff',
    hoverBg: '#333333',
  },
};

export const defaultThemeConfig: ThemeConfig = {
  mode: 'light',
  reducedMotion: false,
  highContrast: false,
  fontSize: 'medium',
  cornerRadius: 'medium',
  colorBlindFriendly: false,
  fontFamily: 'inter',
};
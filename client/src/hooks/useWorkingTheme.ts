import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Research-backed color palettes for neurodivergent users
export const COLOR_PALETTES = {
  // Autism-friendly: Low sensory load, muted tones
  'autism-calm': {
    name: 'Autism Calm',
    description: 'Muted, low-sensory colors to reduce overwhelm',
    primary: '#6B8DB5', // Soft blue
    secondary: '#8FA68E', // Muted green
    accent: '#B5A692', // Warm beige
    background: '#F8F9FA',
    surface: '#FFFFFF',
    text: '#2C3E50',
    textSecondary: '#5D6D7E',
  },
  
  // ADHD-focused: Higher contrast for better focus
  'adhd-focus': {
    name: 'ADHD Focus',
    description: 'High contrast colors to improve focus and attention',
    primary: '#2563EB', // Bright blue
    secondary: '#7C3AED', // Purple
    accent: '#F59E0B', // Orange for alerts
    background: '#FFFFFF',
    surface: '#F3F4F6',
    text: '#111827',
    textSecondary: '#4B5563',
  },
  
  // Dyslexia-optimized: High contrast, dyslexia-friendly
  'dyslexia-friendly': {
    name: 'Dyslexia Support',
    description: 'High contrast with dyslexia-friendly color combinations',
    primary: '#1F2937', // Dark gray
    secondary: '#4F46E5', // Indigo
    accent: '#EF4444', // Red for important items
    background: '#FEF3C7', // Cream background (easier on eyes)
    surface: '#FFFFFF',
    text: '#111827',
    textSecondary: '#374151',
  },
  
  // Sensory minimal: Very low stimulation
  'sensory-minimal': {
    name: 'Sensory Minimal',
    description: 'Minimal sensory load with neutral tones',
    primary: '#64748B', // Neutral gray
    secondary: '#94A3B8', // Light gray
    accent: '#475569', // Darker gray
    background: '#F8FAFC',
    surface: '#FFFFFF',
    text: '#1E293B',
    textSecondary: '#64748B',
  },
  
  // High contrast for visual accessibility
  'high-contrast': {
    name: 'High Contrast',
    description: 'Maximum contrast for better visibility',
    primary: '#000000',
    secondary: '#1F2937',
    accent: '#DC2626',
    background: '#FFFFFF',
    surface: '#F9FAFB',
    text: '#000000',
    textSecondary: '#374151',
  },
  
  // Ocean calm: Cool, soothing tones
  'ocean-calm': {
    name: 'Ocean Calm',
    description: 'Cool blues and greens for a calming effect',
    primary: '#0EA5E9', // Sky blue
    secondary: '#06B6D4', // Cyan
    accent: '#10B981', // Emerald
    background: '#F0F9FF',
    surface: '#FFFFFF',
    text: '#0C4A6E',
    textSecondary: '#0369A1',
  },
  
  // Forest peace: Warm, earthy tones
  'forest-peace': {
    name: 'Forest Peace',
    description: 'Warm earth tones for grounding and comfort',
    primary: '#059669', // Green
    secondary: '#D97706', // Amber
    accent: '#DC2626', // Red accent
    background: '#F0FDF4',
    surface: '#FFFFFF',
    text: '#064E3B',
    textSecondary: '#047857',
  },
  
  // Sunset warm: Warm, energizing colors
  'sunset-warm': {
    name: 'Sunset Warm',
    description: 'Warm colors to boost energy and mood',
    primary: '#EA580C', // Orange
    secondary: '#DC2626', // Red
    accent: '#F59E0B', // Amber
    background: '#FEF3C7',
    surface: '#FFFFFF',
    text: '#7C2D12',
    textSecondary: '#EA580C',
  },
};

export interface ThemeState {
  // Core settings
  colorPalette: keyof typeof COLOR_PALETTES;
  darkMode: boolean;
  reducedMotion: boolean;
  highContrast: boolean;
  
  // Typography
  fontSize: number; // in rem
  fontFamily: 'inter' | 'poppins' | 'opendyslexic' | 'comic-sans';
  lineHeight: number;
  letterSpacing: number;
  
  // Layout
  cornerRadius: number; // in px
  spacing: number; // multiplier
  
  // Accessibility
  focusRingVisible: boolean;
  soundEnabled: boolean;
  
  // Neurodivergent-specific features
  flashingDisabled: boolean;
  autoplayDisabled: boolean;
  
  // Actions
  setColorPalette: (palette: keyof typeof COLOR_PALETTES) => void;
  setDarkMode: (enabled: boolean) => void;
  setReducedMotion: (enabled: boolean) => void;
  setHighContrast: (enabled: boolean) => void;
  setFontSize: (size: number) => void;
  setFontFamily: (family: ThemeState['fontFamily']) => void;
  setLineHeight: (height: number) => void;
  setLetterSpacing: (spacing: number) => void;
  setCornerRadius: (radius: number) => void;
  setSpacing: (spacing: number) => void;
  setFocusRingVisible: (visible: boolean) => void;
  setSoundEnabled: (enabled: boolean) => void;
  setFlashingDisabled: (disabled: boolean) => void;
  setAutoplayDisabled: (disabled: boolean) => void;
  applyPreset: (preset: 'autism-friendly' | 'adhd-focused' | 'dyslexia-optimized' | 'sensory-minimal') => void;
  resetToDefaults: () => void;
  exportSettings: () => string;
  importSettings: (settings: string) => void;
}

const defaultTheme: Omit<ThemeState, keyof ThemeActions> = {
  colorPalette: 'ocean-calm',
  darkMode: false,
  reducedMotion: false,
  highContrast: false,
  fontSize: 1, // 1rem = 16px
  fontFamily: 'inter',
  lineHeight: 1.5,
  letterSpacing: 0,
  cornerRadius: 8,
  spacing: 1,
  focusRingVisible: true,
  soundEnabled: false,
  flashingDisabled: true,
  autoplayDisabled: true,
};

type ThemeActions = {
  setColorPalette: (palette: keyof typeof COLOR_PALETTES) => void;
  setDarkMode: (enabled: boolean) => void;
  setReducedMotion: (enabled: boolean) => void;
  setHighContrast: (enabled: boolean) => void;
  setFontSize: (size: number) => void;
  setFontFamily: (family: ThemeState['fontFamily']) => void;
  setLineHeight: (height: number) => void;
  setLetterSpacing: (spacing: number) => void;
  setCornerRadius: (radius: number) => void;
  setSpacing: (spacing: number) => void;
  setFocusRingVisible: (visible: boolean) => void;
  setSoundEnabled: (enabled: boolean) => void;
  setFlashingDisabled: (disabled: boolean) => void;
  setAutoplayDisabled: (disabled: boolean) => void;
  applyPreset: (preset: 'autism-friendly' | 'adhd-focused' | 'dyslexia-optimized' | 'sensory-minimal') => void;
  resetToDefaults: () => void;
  exportSettings: () => string;
  importSettings: (settings: string) => void;
};

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      ...defaultTheme,
      
      setColorPalette: (palette) => set({ colorPalette: palette }),
      setDarkMode: (darkMode) => set({ darkMode }),
      setReducedMotion: (reducedMotion) => set({ reducedMotion }),
      setHighContrast: (highContrast) => set({ highContrast }),
      setFontSize: (fontSize) => set({ fontSize }),
      setFontFamily: (fontFamily) => set({ fontFamily }),
      setLineHeight: (lineHeight) => set({ lineHeight }),
      setLetterSpacing: (letterSpacing) => set({ letterSpacing }),
      setCornerRadius: (cornerRadius) => set({ cornerRadius }),
      setSpacing: (spacing) => set({ spacing }),
      setFocusRingVisible: (focusRingVisible) => set({ focusRingVisible }),
      setSoundEnabled: (soundEnabled) => set({ soundEnabled }),
      setFlashingDisabled: (flashingDisabled) => set({ flashingDisabled }),
      setAutoplayDisabled: (autoplayDisabled) => set({ autoplayDisabled }),
      
      applyPreset: (preset) => {
        const presets = {
          'autism-friendly': {
            colorPalette: 'autism-calm' as keyof typeof COLOR_PALETTES,
            reducedMotion: true,
            fontSize: 1.1,
            spacing: 1.2,
            cornerRadius: 12,
            flashingDisabled: true,
            soundEnabled: false,
          },
          'adhd-focused': {
            colorPalette: 'adhd-focus' as keyof typeof COLOR_PALETTES,
            highContrast: true,
            fontSize: 1.1,
            spacing: 1,
            cornerRadius: 6,
            focusRingVisible: true,
          },
          'dyslexia-optimized': {
            colorPalette: 'dyslexia-friendly' as keyof typeof COLOR_PALETTES,
            fontFamily: 'opendyslexic' as ThemeState['fontFamily'],
            fontSize: 1.2,
            lineHeight: 1.7,
            letterSpacing: 0.5,
            spacing: 1.3,
          },
          'sensory-minimal': {
            colorPalette: 'sensory-minimal' as keyof typeof COLOR_PALETTES,
            reducedMotion: true,
            flashingDisabled: true,
            soundEnabled: false,
            cornerRadius: 4,
            spacing: 1.1,
          },
        };
        
        set({ ...presets[preset] });
      },
      
      resetToDefaults: () => set(defaultTheme),
      
      exportSettings: () => {
        const state = get();
        const { exportSettings, importSettings, applyPreset, resetToDefaults, ...settings } = state;
        return JSON.stringify(settings, null, 2);
      },
      
      importSettings: (settingsJson) => {
        try {
          const settings = JSON.parse(settingsJson);
          set(settings);
        } catch (error) {
          console.error('Failed to import settings:', error);
        }
      },
    }),
    {
      name: 'neuroflow-theme',
      version: 1,
    }
  )
);

// CSS variable generator
export const generateCSSVariables = (state: ThemeState) => {
  const palette = COLOR_PALETTES[state.colorPalette];
  const fontFamilies = {
    inter: '"Inter", system-ui, -apple-system, sans-serif',
    poppins: '"Poppins", system-ui, -apple-system, sans-serif',
    opendyslexic: '"OpenDyslexic", monospace, sans-serif',
    'comic-sans': '"Comic Sans MS", cursive, sans-serif',
  };
  
  return {
    '--color-primary': palette.primary,
    '--color-secondary': palette.secondary,
    '--color-accent': palette.accent,
    '--color-background': state.darkMode ? '#0F172A' : palette.background,
    '--color-surface': state.darkMode ? '#1E293B' : palette.surface,
    '--color-text': state.darkMode ? '#F8FAFC' : palette.text,
    '--color-text-secondary': state.darkMode ? '#CBD5E1' : palette.textSecondary,
    '--font-size-base': `${state.fontSize}rem`,
    '--font-family': fontFamilies[state.fontFamily],
    '--line-height': state.lineHeight.toString(),
    '--letter-spacing': `${state.letterSpacing}px`,
    '--border-radius': `${state.cornerRadius}px`,
    '--spacing-unit': `${state.spacing}rem`,
    '--focus-ring-width': state.focusRingVisible ? '2px' : '0px',
    '--animation-duration': state.reducedMotion ? '0.01ms' : '200ms',
  };
};
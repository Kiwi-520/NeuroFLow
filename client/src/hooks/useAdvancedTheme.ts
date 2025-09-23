import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Enhanced theme interfaces for comprehensive customization
export interface AccessibilitySettings {
  reducedMotion: boolean;
  highContrast: boolean;
  focusRingEnabled: boolean;
  screenReaderOptimized: boolean;
  keyboardNavigationEnhanced: boolean;
  autoplayDisabled: boolean;
}

export interface VisualSettings {
  fontSize: 'xs' | 'sm' | 'base' | 'lg' | 'xl' | '2xl';
  fontFamily: 'inter' | 'poppins' | 'openDyslexic' | 'system';
  lineHeight: 'tight' | 'normal' | 'relaxed' | 'loose';
  letterSpacing: 'tight' | 'normal' | 'wide';
  cornerRadius: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'full';
  spacing: 'compact' | 'normal' | 'comfortable' | 'spacious';
}

export interface ColorSettings {
  mode: 'light' | 'dark' | 'auto';
  colorBlindFriendly: boolean;
  saturation: 'low' | 'normal' | 'high';
  warmth: 'cool' | 'neutral' | 'warm';
  customPrimaryColor?: string;
  customAccentColor?: string;
}

export interface AnimationSettings {
  enableAnimations: boolean;
  animationSpeed: 'slow' | 'normal' | 'fast';
  parallaxEffects: boolean;
  hoverEffects: boolean;
  transitionStyle: 'smooth' | 'gentle' | 'crisp';
}

export interface SensorySettings {
  soundEnabled: boolean;
  hapticFeedback: boolean;
  flashingEffectsDisabled: boolean;
  autoContrastAdjustment: boolean;
  blueLightFilter: boolean;
  backgroundTextures: 'none' | 'subtle' | 'prominent';
}

export interface PersonalizationSettings {
  dailyMoodTracking: boolean;
  adaptiveInterface: boolean;
  contextualHelp: boolean;
  personalizedPrompts: boolean;
  learningProgress: boolean;
  favoriteColors: string[];
}

export interface ThemeState {
  // Settings groups
  accessibility: AccessibilitySettings;
  visual: VisualSettings;
  color: ColorSettings;
  animation: AnimationSettings;
  sensory: SensorySettings;
  personalization: PersonalizationSettings;
  
  // Derived state
  computedTheme: ComputedTheme;
  
  // Actions
  updateAccessibility: (settings: Partial<AccessibilitySettings>) => void;
  updateVisual: (settings: Partial<VisualSettings>) => void;
  updateColor: (settings: Partial<ColorSettings>) => void;
  updateAnimation: (settings: Partial<AnimationSettings>) => void;
  updateSensory: (settings: Partial<SensorySettings>) => void;
  updatePersonalization: (settings: Partial<PersonalizationSettings>) => void;
  resetToDefaults: () => void;
  exportSettings: () => string;
  importSettings: (settingsJson: string) => void;
  applyPreset: (preset: ThemePreset) => void;
}

export interface ComputedTheme {
  cssVariables: Record<string, string>;
  classNames: string[];
  mediaQueries: Record<string, string>;
}

export type ThemePreset = 
  | 'autism-friendly'
  | 'adhd-focused'
  | 'dyslexia-optimized'
  | 'sensory-minimal'
  | 'high-contrast'
  | 'gentle-pastels'
  | 'earth-tones'
  | 'ocean-calm';

// Default settings optimized for neurodivergent users
const defaultSettings = {
  accessibility: {
    reducedMotion: false,
    highContrast: false,
    focusRingEnabled: true,
    screenReaderOptimized: false,
    keyboardNavigationEnhanced: true,
    autoplayDisabled: true,
  } as AccessibilitySettings,
  
  visual: {
    fontSize: 'base' as const,
    fontFamily: 'inter' as const,
    lineHeight: 'normal' as const,
    letterSpacing: 'normal' as const,
    cornerRadius: 'md' as const,
    spacing: 'normal' as const,
  } as VisualSettings,
  
  color: {
    mode: 'light' as const,
    colorBlindFriendly: false,
    saturation: 'normal' as const,
    warmth: 'neutral' as const,
  } as ColorSettings,
  
  animation: {
    enableAnimations: true,
    animationSpeed: 'normal' as const,
    parallaxEffects: false,
    hoverEffects: true,
    transitionStyle: 'gentle' as const,
  } as AnimationSettings,
  
  sensory: {
    soundEnabled: false,
    hapticFeedback: false,
    flashingEffectsDisabled: true,
    autoContrastAdjustment: false,
    blueLightFilter: false,
    backgroundTextures: 'subtle' as const,
  } as SensorySettings,
  
  personalization: {
    dailyMoodTracking: true,
    adaptiveInterface: true,
    contextualHelp: true,
    personalizedPrompts: true,
    learningProgress: true,
    favoriteColors: ['#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b'],
  } as PersonalizationSettings,
};

// Theme computation function
const computeTheme = (settings: {
  accessibility: AccessibilitySettings;
  visual: VisualSettings;
  color: ColorSettings;
  animation: AnimationSettings;
  sensory: SensorySettings;
  personalization: PersonalizationSettings;
}): ComputedTheme => {
  const cssVariables: Record<string, string> = {};
  const classNames: string[] = [];
  
  // Apply color mode
  classNames.push(settings.color.mode === 'dark' ? 'dark' : 'light');
  
  // Apply accessibility settings
  if (settings.accessibility.reducedMotion) {
    classNames.push('reduce-motion');
  }
  if (settings.accessibility.highContrast) {
    classNames.push('high-contrast');
  }
  
  // Apply visual settings
  cssVariables['--font-size-base'] = {
    'xs': '0.75rem',
    'sm': '0.875rem',
    'base': '1rem',
    'lg': '1.125rem',
    'xl': '1.25rem',
    '2xl': '1.5rem',
  }[settings.visual.fontSize];
  
  cssVariables['--font-family-primary'] = {
    'inter': 'Inter, system-ui, sans-serif',
    'poppins': 'Poppins, system-ui, sans-serif',
    'openDyslexic': 'OpenDyslexic, monospace',
    'system': 'system-ui, -apple-system, sans-serif',
  }[settings.visual.fontFamily];
  
  cssVariables['--line-height-base'] = {
    'tight': '1.25',
    'normal': '1.5',
    'relaxed': '1.625',
    'loose': '2',
  }[settings.visual.lineHeight];
  
  cssVariables['--border-radius'] = {
    'none': '0px',
    'sm': '0.25rem',
    'md': '0.5rem',
    'lg': '0.75rem',
    'xl': '1rem',
    'full': '9999px',
  }[settings.visual.cornerRadius];
  
  // Apply animation settings
  if (!settings.animation.enableAnimations) {
    classNames.push('no-animations');
  }
  
  cssVariables['--animation-duration'] = {
    'slow': '0.5s',
    'normal': '0.25s',
    'fast': '0.15s',
  }[settings.animation.animationSpeed];
  
  return {
    cssVariables,
    classNames,
    mediaQueries: {},
  };
};

// Zustand store
export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      ...defaultSettings,
      computedTheme: computeTheme(defaultSettings),
      
      updateAccessibility: (newSettings) => set((state) => {
        const accessibility = { ...state.accessibility, ...newSettings };
        const newState = { ...state, accessibility };
        return { ...newState, computedTheme: computeTheme(newState) };
      }),
      
      updateVisual: (newSettings) => set((state) => {
        const visual = { ...state.visual, ...newSettings };
        const newState = { ...state, visual };
        return { ...newState, computedTheme: computeTheme(newState) };
      }),
      
      updateColor: (newSettings) => set((state) => {
        const color = { ...state.color, ...newSettings };
        const newState = { ...state, color };
        return { ...newState, computedTheme: computeTheme(newState) };
      }),
      
      updateAnimation: (newSettings) => set((state) => {
        const animation = { ...state.animation, ...newSettings };
        const newState = { ...state, animation };
        return { ...newState, computedTheme: computeTheme(newState) };
      }),
      
      updateSensory: (newSettings) => set((state) => {
        const sensory = { ...state.sensory, ...newSettings };
        const newState = { ...state, sensory };
        return { ...newState, computedTheme: computeTheme(newState) };
      }),
      
      updatePersonalization: (newSettings) => set((state) => {
        const personalization = { ...state.personalization, ...newSettings };
        const newState = { ...state, personalization };
        return { ...newState, computedTheme: computeTheme(newState) };
      }),
      
      resetToDefaults: () => set({
        ...defaultSettings,
        computedTheme: computeTheme(defaultSettings),
      }),
      
      exportSettings: () => {
        const { computedTheme, ...settings } = get();
        return JSON.stringify(settings, null, 2);
      },
      
      importSettings: (settingsJson) => {
        try {
          const settings = JSON.parse(settingsJson);
          const newState = { ...settings };
          set({ ...newState, computedTheme: computeTheme(newState) });
        } catch (error) {
          console.error('Failed to import settings:', error);
        }
      },
      
      applyPreset: (preset) => {
        const presets: Record<ThemePreset, Partial<ThemeState>> = {
          'autism-friendly': {
            accessibility: { ...defaultSettings.accessibility, reducedMotion: true },
            visual: { ...defaultSettings.visual, spacing: 'comfortable' },
            color: { ...defaultSettings.color, saturation: 'low' },
            sensory: { ...defaultSettings.sensory, flashingEffectsDisabled: true, backgroundTextures: 'none' },
          },
          'adhd-focused': {
            visual: { ...defaultSettings.visual, spacing: 'compact', fontSize: 'lg' },
            color: { ...defaultSettings.color, saturation: 'high' },
            animation: { ...defaultSettings.animation, animationSpeed: 'fast' },
          },
          'dyslexia-optimized': {
            visual: { 
              ...defaultSettings.visual, 
              fontFamily: 'openDyslexic', 
              lineHeight: 'relaxed',
              letterSpacing: 'wide',
              fontSize: 'lg'
            },
          },
          'sensory-minimal': {
            accessibility: { ...defaultSettings.accessibility, reducedMotion: true },
            color: { ...defaultSettings.color, saturation: 'low' },
            sensory: { 
              ...defaultSettings.sensory, 
              backgroundTextures: 'none',
              flashingEffectsDisabled: true 
            },
            animation: { ...defaultSettings.animation, enableAnimations: false },
          },
          'high-contrast': {
            accessibility: { ...defaultSettings.accessibility, highContrast: true },
            color: { ...defaultSettings.color, saturation: 'high' },
          },
          'gentle-pastels': {
            color: { ...defaultSettings.color, saturation: 'low', warmth: 'warm' },
            animation: { ...defaultSettings.animation, transitionStyle: 'gentle' },
          },
          'earth-tones': {
            color: { ...defaultSettings.color, warmth: 'warm' },
            sensory: { ...defaultSettings.sensory, backgroundTextures: 'subtle' },
          },
          'ocean-calm': {
            color: { ...defaultSettings.color, warmth: 'cool', saturation: 'low' },
            animation: { ...defaultSettings.animation, transitionStyle: 'smooth' },
          },
        };
        
        const presetSettings = presets[preset];
        if (presetSettings) {
          const currentState = get();
          const newState = {
            ...currentState,
            ...presetSettings,
          };
          set({ ...newState, computedTheme: computeTheme(newState) });
        }
      },
    }),
    {
      name: 'neuroflow-theme-advanced',
      version: 1,
    }
  )
);
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { 
  ThemeConfig, 
  ThemeColors, 
  lightTheme, 
  darkTheme, 
  highContrastLight, 
  highContrastDark,
  defaultThemeConfig 
} from '../styles/theme';

interface ThemeStore {
  config: ThemeConfig;
  colors: ThemeColors;
  
  // Actions
  setMode: (mode: ThemeConfig['mode']) => void;
  setReducedMotion: (reduced: boolean) => void;
  setHighContrast: (enabled: boolean) => void;
  setFontSize: (size: ThemeConfig['fontSize']) => void;
  setCornerRadius: (radius: ThemeConfig['cornerRadius']) => void;
  setColorBlindFriendly: (enabled: boolean) => void;
  setFontFamily: (family: ThemeConfig['fontFamily']) => void;
  resetToDefaults: () => void;
}

const getThemeColors = (config: ThemeConfig): ThemeColors => {
  const isDark = config.mode === 'dark' || 
    (config.mode === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  
  if (config.highContrast) {
    return isDark ? highContrastDark : highContrastLight;
  }
  
  return isDark ? darkTheme : lightTheme;
};

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set, get) => ({
      config: defaultThemeConfig,
      colors: lightTheme,
      
      setMode: (mode) => set((state) => {
        const newConfig = { ...state.config, mode };
        return {
          config: newConfig,
          colors: getThemeColors(newConfig),
        };
      }),
      
      setReducedMotion: (reducedMotion) => set((state) => ({
        config: { ...state.config, reducedMotion },
      })),
      
      setHighContrast: (highContrast) => set((state) => {
        const newConfig = { ...state.config, highContrast };
        return {
          config: newConfig,
          colors: getThemeColors(newConfig),
        };
      }),
      
      setFontSize: (fontSize) => set((state) => ({
        config: { ...state.config, fontSize },
      })),
      
      setCornerRadius: (cornerRadius) => set((state) => ({
        config: { ...state.config, cornerRadius },
      })),
      
      setColorBlindFriendly: (colorBlindFriendly) => set((state) => ({
        config: { ...state.config, colorBlindFriendly },
      })),
      
      setFontFamily: (fontFamily) => set((state) => ({
        config: { ...state.config, fontFamily },
      })),
      
      resetToDefaults: () => set({
        config: defaultThemeConfig,
        colors: lightTheme,
      }),
    }),
    {
      name: 'neuroflow-theme',
      version: 1,
    }
  )
);

// Auto-detect system theme preference
if (typeof window !== 'undefined') {
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  const handleThemeChange = () => {
    const { config } = useThemeStore.getState();
    if (config.mode === 'auto') {
      useThemeStore.setState({
        colors: getThemeColors(config),
      });
    }
  };
  
  mediaQuery.addEventListener('change', handleThemeChange);
}

// Auto-detect reduced motion preference
if (typeof window !== 'undefined') {
  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (motionQuery.matches) {
    useThemeStore.getState().setReducedMotion(true);
  }
}
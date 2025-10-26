import React, { useEffect } from 'react';
import { useThemeStore, generateCSSVariables } from '../hooks/useWorkingTheme';

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const WorkingThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const themeState = useThemeStore();

  useEffect(() => {
    // Apply CSS variables to the root element
    const root = document.documentElement;
    const cssVariables = generateCSSVariables(themeState);
    
    Object.entries(cssVariables).forEach(([property, value]) => {
      root.style.setProperty(property, value as string);
    });

    // Apply global theme styles to body
    const body = document.body;
    
    // Apply font styles with !important to ensure they override everything
    body.style.setProperty('font-family', cssVariables['--font-family'] as string, 'important');
    body.style.setProperty('font-size', cssVariables['--font-size-base'] as string, 'important');
    body.style.setProperty('line-height', cssVariables['--line-height'] as string, 'important');
    body.style.setProperty('letter-spacing', cssVariables['--letter-spacing'] as string, 'important');
    body.style.setProperty('color', cssVariables['--color-text'] as string, 'important');
    body.style.setProperty('background-color', cssVariables['--color-background'] as string, 'important');

    // Apply theme classes to body
    body.classList.remove(
      'dark-mode',
      'high-contrast', 
      'reduced-motion',
      'no-focus-ring',
      'sound-enabled',
      'flashing-disabled',
      'autoplay-disabled'
    );

    // Add current theme classes
    if (themeState.darkMode) body.classList.add('dark-mode');
    if (themeState.highContrast) body.classList.add('high-contrast');
    if (themeState.reducedMotion) body.classList.add('reduced-motion');
    if (!themeState.focusRingVisible) body.classList.add('no-focus-ring');
    if (themeState.soundEnabled) body.classList.add('sound-enabled');
    if (themeState.flashingDisabled) body.classList.add('flashing-disabled');
    if (themeState.autoplayDisabled) body.classList.add('autoplay-disabled');

    // Set color scheme for better OS integration
    root.style.colorScheme = themeState.darkMode ? 'dark' : 'light';

    // Handle reduced motion preference
    if (themeState.reducedMotion) {
      root.style.setProperty('--animation-duration', '0.01ms');
      root.style.setProperty('--transition-duration', '0.01ms');
    } else {
      root.style.setProperty('--animation-duration', '200ms');
      root.style.setProperty('--transition-duration', '150ms');
    }

    // Add meta tag for mobile browsers
    let metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (!metaThemeColor) {
      metaThemeColor = document.createElement('meta');
      metaThemeColor.setAttribute('name', 'theme-color');
      document.head.appendChild(metaThemeColor);
    }
    metaThemeColor.setAttribute('content', cssVariables['--color-primary'] as string);

  }, [themeState]);

  return <>{children}</>;
};
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
      root.style.setProperty(property, value);
    });

    // Apply font family to body
    document.body.style.fontFamily = `var(--font-family)`;
    document.body.style.fontSize = `var(--font-size-base)`;
    document.body.style.lineHeight = `var(--line-height)`;
    document.body.style.letterSpacing = `var(--letter-spacing)`;

    // Apply theme classes to body
    const body = document.body;
    
    // Remove all theme classes first
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

    // Apply high contrast if enabled
    if (themeState.highContrast) {
      root.style.setProperty('--color-text', '#000000');
      root.style.setProperty('--color-background', '#FFFFFF');
      root.style.setProperty('--color-surface', '#F3F4F6');
    }

    // Handle reduced motion preference
    if (themeState.reducedMotion) {
      root.style.setProperty('--animation-duration', '0.01ms');
      root.style.setProperty('--transition-duration', '0.01ms');
    }

    // Add meta tag for mobile browsers
    let metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (!metaThemeColor) {
      metaThemeColor = document.createElement('meta');
      metaThemeColor.setAttribute('name', 'theme-color');
      document.head.appendChild(metaThemeColor);
    }
    metaThemeColor.setAttribute('content', cssVariables['--color-primary']);

  }, [themeState]);

  return <>{children}</>;
};
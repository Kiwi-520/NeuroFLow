import React, { useEffect, ReactNode } from 'react';
import { useThemeStore } from '../hooks/useAdvancedTheme';

interface AdvancedThemeProviderProps {
  children: ReactNode;
}

export const AdvancedThemeProvider: React.FC<AdvancedThemeProviderProps> = ({ children }) => {
  const { computedTheme, accessibility, color, animation, updateColor, updateAccessibility } = useThemeStore();

  useEffect(() => {
    // Apply computed theme to document
    const root = document.documentElement;

    // Apply CSS variables
    Object.entries(computedTheme.cssVariables).forEach(([key, value]) => {
      root.style.setProperty(key, value);
    });

    // Apply class names
    // Remove existing theme classes
    root.classList.remove('light', 'dark', 'high-contrast', 'reduce-motion', 'no-animations');
    
    // Add new theme classes
    computedTheme.classNames.forEach((className) => {
      root.classList.add(className);
    });

    // Handle accessibility preferences
    if (accessibility.reducedMotion) {
      root.style.setProperty('--animation-duration', '0.01ms');
      root.style.setProperty('--transition-duration', '0.01ms');
    }

  }, [computedTheme, accessibility]);

  useEffect(() => {
    // Auto-detect system preferences
    const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)');

    const handleDarkModeChange = (e: MediaQueryListEvent) => {
      if (color.mode === 'auto') {
        updateColor({ mode: e.matches ? 'dark' : 'light' });
      }
    };

    const handleReducedMotionChange = (e: MediaQueryListEvent) => {
      updateAccessibility({ reducedMotion: e.matches });
    };

    const handleHighContrastChange = (e: MediaQueryListEvent) => {
      updateAccessibility({ highContrast: e.matches });
    };

    // Set initial values based on system preferences
    if (color.mode === 'auto') {
      updateColor({ mode: darkModeQuery.matches ? 'dark' : 'light' });
    }

    if (reducedMotionQuery.matches) {
      updateAccessibility({ reducedMotion: true });
    }

    if (highContrastQuery.matches) {
      updateAccessibility({ highContrast: true });
    }

    // Add event listeners
    darkModeQuery.addEventListener('change', handleDarkModeChange);
    reducedMotionQuery.addEventListener('change', handleReducedMotionChange);
    highContrastQuery.addEventListener('change', handleHighContrastChange);

    // Cleanup
    return () => {
      darkModeQuery.removeEventListener('change', handleDarkModeChange);
      reducedMotionQuery.removeEventListener('change', handleReducedMotionChange);
      highContrastQuery.removeEventListener('change', handleHighContrastChange);
    };
  }, [color.mode, updateColor, updateAccessibility]);

  useEffect(() => {
    // Add global styles for better accessibility
    const style = document.createElement('style');
    style.textContent = `
      /* Enhanced focus rings for accessibility */
      *:focus-visible {
        outline: 2px solid var(--color-primary-500, #0ea5e9);
        outline-offset: 2px;
        border-radius: var(--border-radius, 0.375rem);
      }

      /* High contrast mode adjustments */
      .high-contrast {
        --shadow-gentle: 0 0 0 1px currentColor;
        --shadow-gentle-lg: 0 0 0 2px currentColor;
      }

      .high-contrast img,
      .high-contrast video {
        filter: contrast(1.2);
      }

      /* Reduced motion styles */
      .reduce-motion *,
      .reduce-motion *::before,
      .reduce-motion *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
      }

      /* No animations class */
      .no-animations * {
        animation: none !important;
        transition: none !important;
      }

      /* Improved text readability */
      body {
        font-family: var(--font-family-primary, 'Inter', system-ui, sans-serif);
        font-size: var(--font-size-base, 1rem);
        line-height: var(--line-height-base, 1.5);
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }

      /* Enhanced button accessibility */
      button, 
      [role="button"] {
        min-height: 44px;
        min-width: 44px;
        touch-action: manipulation;
      }

      /* Skip links for screen readers */
      .skip-link {
        position: absolute;
        top: -40px;
        left: 6px;
        background: var(--color-primary-600, #0284c7);
        color: white;
        padding: 8px 16px;
        border-radius: var(--border-radius, 0.375rem);
        text-decoration: none;
        z-index: 9999;
        transition: top 0.3s ease;
      }

      .skip-link:focus {
        top: 6px;
      }

      /* Error and success states */
      .error-state {
        border-color: #ef4444;
        background-color: #fef2f2;
      }

      .success-state {
        border-color: #10b981;
        background-color: #f0fdf4;
      }

      /* Dark mode adjustments */
      .dark .error-state {
        background-color: #1f1f1f;
        border-color: #dc2626;
      }

      .dark .success-state {
        background-color: #0a0a0a;
        border-color: #059669;
      }
    `;

    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  return (
    <>
      {/* Skip link for accessibility */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <div id="main-content">
        {children}
      </div>
    </>
  );
};
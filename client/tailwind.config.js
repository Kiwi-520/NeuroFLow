/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Theme-aware colors using CSS variables
        primary: 'var(--color-primary)',
        secondary: 'var(--color-secondary)',
        accent: 'var(--color-accent)',
        background: 'var(--color-background)',
        surface: 'var(--color-surface)',
        'text-primary': 'var(--color-text)',
        'text-secondary': 'var(--color-text-secondary)',
        border: 'var(--color-border)',
        hover: 'var(--color-hover)',
        
        // Keep some static colors for specific use cases
        emotion: {
          calm: '#bfdbfe',
          focused: '#c7d2fe',
          energized: '#fed7aa',
          relaxed: '#bbf7d0',
          positive: '#fef3c7',
          warning: '#fed7aa',
          error: '#fecaca',
          success: '#bbf7d0',
        },
      },
      fontFamily: {
        'theme': 'var(--font-family)',
        inter: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        poppins: ['Poppins', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
      fontSize: {
        'theme-xs': ['calc(var(--font-size-base) * 0.75)', { lineHeight: 'var(--line-height)' }],
        'theme-sm': ['calc(var(--font-size-base) * 0.875)', { lineHeight: 'var(--line-height)' }],
        'theme-base': ['var(--font-size-base)', { lineHeight: 'var(--line-height)' }],
        'theme-lg': ['calc(var(--font-size-base) * 1.125)', { lineHeight: 'var(--line-height)' }],
        'theme-xl': ['calc(var(--font-size-base) * 1.25)', { lineHeight: 'var(--line-height)' }],
        'theme-2xl': ['calc(var(--font-size-base) * 1.5)', { lineHeight: 'var(--line-height)' }],
        'theme-3xl': ['calc(var(--font-size-base) * 1.875)', { lineHeight: 'var(--line-height)' }],
        'theme-4xl': ['calc(var(--font-size-base) * 2.25)', { lineHeight: 'var(--line-height)' }],
      },
      borderRadius: {
        'theme': 'var(--border-radius)',
      },
      spacing: {
        'theme-xs': 'var(--spacing-xs)',
        'theme-sm': 'var(--spacing-sm)', 
        'theme-md': 'var(--spacing-md)',
        'theme-lg': 'var(--spacing-lg)',
        'theme-xl': 'var(--spacing-xl)',
      },
      animation: {
        'gentle-fade-in': 'gentleFadeIn 0.25s ease-out',
        'gentle-slide-up': 'gentleSlideUp 0.25s ease-out',
        'gentle-scale': 'gentleScale 0.15s ease-out',
      },
      keyframes: {
        gentleFadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        gentleSlideUp: {
          '0%': { opacity: '0', transform: 'translateY(1rem)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        gentleScale: {
          '0%': { transform: 'scale(0.95)' },
          '100%': { transform: 'scale(1)' },
        },
      },
      boxShadow: {
        'gentle': '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
        'gentle-lg': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
      },
    },
  },
  plugins: [],
}
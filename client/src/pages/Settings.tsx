import React from 'react';import React from 'react';

import { useThemeStore } from '../hooks/useWorkingTheme';import { useThemeStore } from '../hooks/useWorkingTheme';



const Settings: React.FC = () => {const Settings: React.FC = () => {

  const themeStore = useThemeStore();  const themeStore = useThemeStore();



  const presets = [  const quickPresets = [

    {    {

      id: 'autism-friendly',      id: 'autism-friendly',

      name: 'Autism Friendly',      name: 'Autism Friendly',

      description: 'Reduced motion, calm colors',      description: 'Reduced motion, calm colors',

      icon: '🔔',      icon: '🔔',

      bgColor: 'bg-orange-50 hover:bg-orange-100',    },

      iconBg: 'bg-orange-100',    {

      borderColor: 'hover:border-orange-200'      id: 'adhd-focused',

    },      name: 'ADHD Focused',

    {      description: 'High contrast, faster animations',

      id: 'adhd-focused',      icon: '⚡',

      name: 'ADHD Focused',     },

      description: 'High contrast, faster animations',    {

      icon: '⚡',      id: 'dyslexia-optimized',

      bgColor: 'bg-yellow-50 hover:bg-yellow-100',      name: 'Dyslexia Support',

      iconBg: 'bg-yellow-100',      description: 'Special font, wider spacing',

      borderColor: 'hover:border-yellow-200'      icon: '📖',

    },    },

    {    {

      id: 'dyslexia-optimized',      id: 'sensory-minimal',

      name: 'Dyslexia Support',      name: 'Sensory Minimal',

      description: 'Special font, wider spacing',      description: 'No animations, muted colors',

      icon: '📖',      icon: '🧊',

      bgColor: 'bg-blue-50 hover:bg-blue-100',     }

      iconBg: 'bg-blue-100',  ];

      borderColor: 'hover:border-blue-200'

    },  const colorPresets = [

    {    {

      id: 'sensory-minimal',      id: 'high-contrast',

      name: 'Sensory Minimal',      name: 'High Contrast',

      description: 'No animations, muted colors',      description: 'Maximum contrast for visibility',

      icon: '🧊',      icon: '�',

      bgColor: 'bg-pink-50 hover:bg-pink-100',    },

      iconBg: 'bg-pink-100',     {

      borderColor: 'hover:border-pink-200'      id: 'gentle-pastels',

    }      name: 'Gentle Pastels',

  ];      description: 'Soft, calming colors',

      icon: '🌸',

  return (    },

    <div className="max-w-7xl mx-auto p-8">    {

      {/* Header */}      id: 'earth-tones',

      <div className="flex items-center gap-3 mb-10">      name: 'Earth Tones',

        <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">      description: 'Natural, grounding colors',

          <span className="text-blue-600 text-lg">✨</span>      icon: '🌿',

        </div>    },

        <h1 className="text-2xl font-bold text-gray-900">Quick Presets</h1>    {

      </div>      id: 'ocean-calm',

      name: 'Ocean Calm',

      {/* Presets Grid - Exactly 4 cards */}      description: 'Cool blues and greens',

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-8">      icon: '🌊',

        {presets.map((preset) => (    }

          <div  ];

            key={preset.id}

            onClick={() => themeStore.applyPreset(preset.id as any)}  return (

            className={`${preset.bgColor} p-8 rounded-3xl border-2 border-transparent ${preset.borderColor} cursor-pointer transition-all duration-300 hover:shadow-lg hover:scale-105 group`}    <div className="container mx-auto px-6 py-8 max-w-6xl">

          >      {/* Quick Presets Section */}

            {/* Icon Container */}      <section className="mb-12">

            <div className={`${preset.iconBg} w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>        <div className="flex items-center gap-3 mb-8">

              <span className="text-3xl">{preset.icon}</span>          <span className="text-2xl">✨</span>

            </div>          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Quick Presets</h2>

                    </div>

            {/* Title */}        

            <h3 className="text-xl font-bold text-gray-900 mb-3 leading-tight">        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

              {preset.name}          {quickPresets.map((preset) => (

            </h3>            <button

                          key={preset.id}

            {/* Description */}              onClick={() => themeStore.applyPreset(preset.id as any)}

            <p className="text-gray-600 text-sm leading-relaxed">              className="group p-6 bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 hover:shadow-lg transition-all duration-200 text-left"

              {preset.description}            >

            </p>              <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-200">

          </div>                {preset.icon}

        ))}              </div>

      </div>              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">

    </div>                {preset.name}

  );              </h3>

};              <p className="text-sm text-gray-600 dark:text-gray-400">

                {preset.description}

export default Settings;              </p>
            </button>
          ))}
        </div>
      </section>

      {/* Color Presets Section */}
      <section>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {colorPresets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => themeStore.setColorPalette(preset.id as any)}
              className="group p-6 bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 hover:shadow-lg transition-all duration-200 text-left"
            >
              <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-200">
                {preset.icon}
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {preset.name}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Settings;
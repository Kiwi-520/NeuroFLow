import React, { useState } from 'react';
import { useThemeStore, COLOR_PALETTES } from '../hooks/useWorkingTheme';
import { 
  Palette, 
  Type, 
  Eye, 
  Accessibility, 
  Volume2, 
  VolumeX,
  Download,
  Upload,
  RotateCcw,
  Zap,
  Focus,
  Brain,
  Heart,
  Save,
  Check,
  X
} from 'lucide-react';

const Settings: React.FC = () => {
  const themeStore = useThemeStore();
  const [activeTab, setActiveTab] = useState('presets');
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importText, setImportText] = useState('');
  const [saveMessage, setSaveMessage] = useState('');
  const [lastPalette, setLastPalette] = useState<keyof typeof COLOR_PALETTES>(themeStore.colorPalette);

  const tabs = [
    { id: 'presets', label: 'Quick Presets', icon: Zap },
    { id: 'colors', label: 'Colors', icon: Palette },
    { id: 'typography', label: 'Typography', icon: Type },
    { id: 'accessibility', label: 'Accessibility', icon: Accessibility },
    { id: 'visual', label: 'Visual', icon: Eye },
    { id: 'personal', label: 'Personal', icon: Heart },
  ];

  const presets = [
    {
      id: 'autism-friendly',
      name: 'Autism Friendly',
      description: 'Reduced motion, calm colors',
      icon: '🔔',
      bgColor: 'bg-orange-50 hover:bg-orange-100',
      iconBg: 'bg-orange-100',
      borderColor: 'hover:border-orange-200',
      selectedBorder: 'border-orange-500',
      selectedBg: 'bg-orange-100'
    },
    {
      id: 'adhd-focused',
      name: 'ADHD Focused', 
      description: 'High contrast, faster animations',
      icon: '⚡',
      bgColor: 'bg-yellow-50 hover:bg-yellow-100',
      iconBg: 'bg-yellow-100',
      borderColor: 'hover:border-yellow-200',
      selectedBorder: 'border-yellow-500',
      selectedBg: 'bg-yellow-100'
    },
    {
      id: 'dyslexia-optimized',
      name: 'Dyslexia Support',
      description: 'Special font, wider spacing',
      icon: '📖',
      bgColor: 'bg-blue-50 hover:bg-blue-100', 
      iconBg: 'bg-blue-100',
      borderColor: 'hover:border-blue-200',
      selectedBorder: 'border-blue-500',
      selectedBg: 'bg-blue-100'
    },
    {
      id: 'sensory-minimal',
      name: 'Sensory Minimal',
      description: 'No animations, muted colors',
      icon: '🧊',
      bgColor: 'bg-pink-50 hover:bg-pink-100',
      iconBg: 'bg-pink-100', 
      borderColor: 'hover:border-pink-200',
      selectedBorder: 'border-pink-500',
      selectedBg: 'bg-pink-100'
    }
  ];

  // Track which preset is currently applied
  const getSelectedPreset = () => {
    // Map color palettes to preset IDs
    const paletteToPreset: Record<string, string> = {
      'autism-calm': 'autism-friendly',
      'adhd-focus': 'adhd-focused', 
      'dyslexia-friendly': 'dyslexia-optimized',
      'sensory-minimal': 'sensory-minimal'
    };
    return paletteToPreset[themeStore.colorPalette] || null;
  };
  const selectedPreset = getSelectedPreset();

  const handlePresetClick = (presetId: string) => {
    themeStore.applyPreset(presetId as any);
    // Update lastPalette to match preset
    const presetToPalette: Record<string, string> = {
      'autism-friendly': 'autism-calm',
      'adhd-focused': 'adhd-focus',
      'dyslexia-optimized': 'dyslexia-friendly',
      'sensory-minimal': 'sensory-minimal',
    };
  setLastPalette((presetToPalette[presetId] as keyof typeof COLOR_PALETTES) || 'pastel-green-default');
  };

  const handleExport = () => {
    const settings = themeStore.exportSettings();
    const blob = new Blob([settings], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'neuroflow-theme.json';
    a.click();
    URL.revokeObjectURL(url);
    setSaveMessage('Settings exported successfully!');
    setTimeout(() => setSaveMessage(''), 3000);
  };

  const handleImport = () => {
    try {
      // Validate JSON before applying
      const parsed = JSON.parse(importText);
      if (!parsed || typeof parsed !== 'object') throw new Error('Invalid format');
      themeStore.importSettings(importText);
      setShowImportDialog(false);
      setImportText('');
      setSaveMessage('Settings imported successfully!');
      setTimeout(() => setSaveMessage(''), 3000);
    } catch (error) {
      setSaveMessage('Failed to import settings. Please check the format.');
      setTimeout(() => setSaveMessage(''), 3000);
    }
  };

  const handleEmergencyReset = () => {
    if (confirm('This will clear all theme settings and reload the page. Are you sure?')) {
      themeStore.emergencyReset();
    }
  };

  const handleFileImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        try {
          JSON.parse(content);
          setImportText(content);
        } catch {
          setSaveMessage('Invalid JSON file.');
          setTimeout(() => setSaveMessage(''), 3000);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
            <span className="text-blue-600 text-lg">⚙️</span>
          </div>
          <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        </div>
        
        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleEmergencyReset}
            className="flex items-center gap-2 px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
            title="Emergency reset - clears all theme data and reloads page"
          >
            <RotateCcw className="w-4 h-4" />
            Emergency Reset
          </button>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
          <button
            onClick={() => setShowImportDialog(true)}
            className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors"
          >
            <Upload className="w-4 h-4" />
            Import
          </button>
          <button
            onClick={themeStore.resetToDefaults}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      {/* Save message */}
      {saveMessage && (
        <div className="mb-6 p-4 bg-green-100 text-green-800 rounded-lg flex items-center gap-2">
          <Check className="w-5 h-5" />
          {saveMessage}
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-8 bg-gray-100 p-1 rounded-lg">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="min-h-96">
        {activeTab === 'presets' && (
          <div>
            <h2 className="text-xl font-semibold mb-6">Quick Presets</h2>
            <p className="text-gray-600 mb-8">
              Choose a preset optimized for different neurodivergent needs. Each preset adjusts multiple settings at once.
            </p>
            
            {/* Presets Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-8">
              {presets.map((preset) => {
                const isSelected = selectedPreset === preset.id;
                return (
                  <div
                    key={preset.id}
                    onClick={() => handlePresetClick(preset.id)}
                    className={`${
                      isSelected ? preset.selectedBg : preset.bgColor
                    } p-8 rounded-3xl border-2 ${
                      isSelected ? preset.selectedBorder : 'border-transparent'
                    } ${preset.borderColor} cursor-pointer transition-all duration-300 hover:shadow-lg hover:scale-105 group ${
                      isSelected ? 'ring-2 ring-blue-300 shadow-lg' : ''
                    }`}
                  >
                    {/* Selected indicator */}
                    {isSelected && (
                      <div className="flex justify-end mb-2">
                        <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                          <span className="text-white text-sm">✓</span>
                        </div>
                      </div>
                    )}
                    
                    {/* Icon Container */}
                    <div className={`${preset.iconBg} w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                      <span className="text-3xl">{preset.icon}</span>
                    </div>
                    
                    {/* Title */}
                    <h3 className="text-xl font-bold text-gray-900 mb-3 leading-tight">
                      {preset.name}
                    </h3>
                    
                    {/* Description */}
                    <p className="text-gray-600 text-sm leading-relaxed">
                      {preset.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === 'colors' && (
          <div>
            <h2 className="text-xl font-semibold mb-6">Color Palette</h2>
            <p className="text-gray-600 mb-8">
              Choose a color palette that works best for your visual needs.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(COLOR_PALETTES).map(([key, palette]) => (
                <div
                  key={key}
                  onClick={() => {
                    themeStore.setColorPalette(key as keyof typeof COLOR_PALETTES);
                    setLastPalette(key as keyof typeof COLOR_PALETTES);
                  }}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                    themeStore.colorPalette === key
                      ? 'border-blue-500 ring-2 ring-blue-200 animate-pulse'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  tabIndex={0}
                  aria-label={`Select ${palette.name} palette`}
                  onKeyDown={e => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      themeStore.setColorPalette(key as keyof typeof COLOR_PALETTES);
                      setLastPalette(key as keyof typeof COLOR_PALETTES);
                    }
                  }}
                >
                  <div className="flex items-center gap-4 mb-3">
                    <div
                      className="w-8 h-8 rounded-full"
                      style={{ backgroundColor: palette.primary }}
                    />
                    <div>
                      <h3 className="font-medium">{palette.name}</h3>
                      <p className="text-sm text-gray-600">{palette.description}</p>
                    </div>
                  </div>
                  
                  {/* Color swatches */}
                  <div className="flex gap-2">
                    <div
                      className="w-6 h-6 rounded-md"
                      style={{ backgroundColor: palette.primary }}
                      title="Primary"
                    />
                    <div
                      className="w-6 h-6 rounded-md"
                      style={{ backgroundColor: palette.secondary }}
                      title="Secondary"
                    />
                    <div
                      className="w-6 h-6 rounded-md"
                      style={{ backgroundColor: palette.accent }}
                      title="Accent"
                    />
                    <div
                      className="w-6 h-6 rounded-md border border-gray-300"
                      style={{ backgroundColor: palette.background }}
                      title="Background"
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Dark Mode Toggle */}
            <div className="mt-8 p-4 bg-gray-50 rounded-lg">
              <label className="flex items-center justify-between">
                <span className="font-medium">Dark Mode</span>
                <button
                  onClick={() => themeStore.setDarkMode(!themeStore.darkMode)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    themeStore.darkMode ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      themeStore.darkMode ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </label>
            </div>
          </div>
        )}

        {activeTab === 'typography' && (
          <div>
            <h2 className="text-xl font-semibold mb-6">Typography</h2>
            <p className="text-gray-600 mb-8">
              Adjust font settings for better readability.
            </p>
            
            <div className="space-y-6">
              {/* Font Family */}
              <div>
                <label className="block text-sm font-medium mb-2">Font Family</label>
                <select
                  value={themeStore.fontFamily}
                  onChange={(e) => themeStore.setFontFamily(e.target.value as any)}
                  className="w-full p-2 border border-gray-300 rounded-md"
                >
                  <option value="inter">Inter (Default)</option>
                  <option value="poppins">Poppins (Friendly)</option>
                  <option value="opendyslexic">OpenDyslexic (Dyslexia-friendly)</option>
                  <option value="comic-sans">Comic Sans (Casual)</option>
                </select>
              </div>

              {/* Font Size */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Font Size: {themeStore.fontSize}rem
                </label>
                <input
                  type="range"
                  min="0.8"
                  max="1.5"
                  step="0.1"
                  value={themeStore.fontSize}
                  onChange={(e) => themeStore.setFontSize(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Small</span>
                  <span>Large</span>
                </div>
              </div>

              {/* Line Height */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Line Height: {themeStore.lineHeight}
                </label>
                <input
                  type="range"
                  min="1.2"
                  max="2"
                  step="0.1"
                  value={themeStore.lineHeight}
                  onChange={(e) => themeStore.setLineHeight(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Tight</span>
                  <span>Loose</span>
                </div>
              </div>

              {/* Letter Spacing */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Letter Spacing: {themeStore.letterSpacing}px
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={themeStore.letterSpacing}
                  onChange={(e) => themeStore.setLetterSpacing(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Normal</span>
                  <span>Wide</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'accessibility' && (
          <div>
            <h2 className="text-xl font-semibold mb-6">Accessibility</h2>
            <p className="text-gray-600 mb-8">
              Adjust settings to improve accessibility and reduce sensory overwhelm.
            </p>
            
            <div className="space-y-6">
              {/* Reduced Motion */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h3 className="font-medium">Reduced Motion</h3>
                  <p className="text-sm text-gray-600">Minimize animations and transitions</p>
                </div>
                <button
                  onClick={() => themeStore.setReducedMotion(!themeStore.reducedMotion)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    themeStore.reducedMotion ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      themeStore.reducedMotion ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* High Contrast */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h3 className="font-medium">High Contrast</h3>
                  <p className="text-sm text-gray-600">Increase contrast for better visibility (Note: May override theme colors)</p>
                </div>
                <button
                  onClick={() => {
                    const newValue = !themeStore.highContrast;
                    themeStore.setHighContrast(newValue);
                    if (!newValue) {
                      // Restore last palette or default
                      themeStore.setColorPalette(lastPalette || 'pastel-green-default');
                    }
                  }}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    themeStore.highContrast ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                  aria-pressed={themeStore.highContrast}
                  tabIndex={0}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      themeStore.highContrast ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* Focus Ring */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h3 className="font-medium">Focus Ring Visible</h3>
                  <p className="text-sm text-gray-600">Show focus indicators for keyboard navigation</p>
                </div>
                <button
                  onClick={() => themeStore.setFocusRingVisible(!themeStore.focusRingVisible)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    themeStore.focusRingVisible ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      themeStore.focusRingVisible ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* Disable Flashing */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h3 className="font-medium">Disable Flashing</h3>
                  <p className="text-sm text-gray-600">Prevent flashing content that may trigger seizures</p>
                </div>
                <button
                  onClick={() => themeStore.setFlashingDisabled(!themeStore.flashingDisabled)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    themeStore.flashingDisabled ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      themeStore.flashingDisabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* Disable Autoplay */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h3 className="font-medium">Disable Autoplay</h3>
                  <p className="text-sm text-gray-600">Prevent videos and audio from playing automatically</p>
                </div>
                <button
                  onClick={() => themeStore.setAutoplayDisabled(!themeStore.autoplayDisabled)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    themeStore.autoplayDisabled ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      themeStore.autoplayDisabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'visual' && (
          <div>
            <h2 className="text-xl font-semibold mb-6">Visual Preferences</h2>
            <p className="text-gray-600 mb-8">
              Customize the visual appearance of the interface.
            </p>
            
            <div className="space-y-6">
              {/* Corner Radius */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Corner Radius: {themeStore.cornerRadius}px
                </label>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="1"
                  value={themeStore.cornerRadius}
                  onChange={(e) => themeStore.setCornerRadius(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Square</span>
                  <span>Rounded</span>
                </div>
              </div>

              {/* Spacing */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Spacing: {themeStore.spacing}x
                </label>
                <input
                  type="range"
                  min="0.8"
                  max="1.5"
                  step="0.1"
                  value={themeStore.spacing}
                  onChange={(e) => themeStore.setSpacing(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Compact</span>
                  <span>Spacious</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'personal' && (
          <div>
            <h2 className="text-xl font-semibold mb-6">Personal Preferences</h2>
            <p className="text-gray-600 mb-8">
              Set your personal preferences for sounds and interactions.
            </p>
            
            <div className="space-y-6">
              {/* Sound Enabled */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  {themeStore.soundEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
                  <div>
                    <h3 className="font-medium">Sound Effects</h3>
                    <p className="text-sm text-gray-600">Enable interface sound effects</p>
                  </div>
                </div>
                <button
                  onClick={() => themeStore.setSoundEnabled(!themeStore.soundEnabled)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    themeStore.soundEnabled ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      themeStore.soundEnabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Import Dialog */}
      {showImportDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Import Settings</h3>
              <button
                onClick={() => setShowImportDialog(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Upload file or paste JSON
                </label>
                <input
                  type="file"
                  accept=".json"
                  onChange={handleFileImport}
                  className="w-full p-2 border border-gray-300 rounded-md mb-2"
                />
                <textarea
                  value={importText}
                  onChange={(e) => setImportText(e.target.value)}
                  placeholder="Paste your settings JSON here..."
                  className="w-full p-2 border border-gray-300 rounded-md"
                  rows={6}
                />
              </div>
              
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setShowImportDialog(false)}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800"
                >
                  Cancel
                </button>
                <button
                  onClick={handleImport}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Import
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Settings;
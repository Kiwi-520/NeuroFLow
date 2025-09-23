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

const FunctionalSettings: React.FC = () => {
  const [activeTab, setActiveTab] = useState('presets');
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importText, setImportText] = useState('');
  const [saveMessage, setSaveMessage] = useState('');
  
  const themeStore = useThemeStore();

  const tabs = [
    { id: 'presets', label: 'Quick Presets', icon: Zap },
    { id: 'colors', label: 'Colors', icon: Palette },
    { id: 'typography', label: 'Typography', icon: Type },
    { id: 'accessibility', label: 'Accessibility', icon: Accessibility },
    { id: 'visual', label: 'Visual', icon: Eye },
    { id: 'personal', label: 'Personal', icon: Heart },
  ];

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

  const applyPreset = (presetName: string) => {
    const presetMap = {
      'Autism-Friendly': 'autism-friendly',
      'ADHD Focus': 'adhd-focused', 
      'Dyslexia Support': 'dyslexia-optimized',
      'Sensory Minimal': 'sensory-minimal',
    } as const;
    
    const presetKey = presetMap[presetName as keyof typeof presetMap];
    if (presetKey) {
      themeStore.applyPreset(presetKey);
      setSaveMessage(`${presetName} preset applied!`);
      setTimeout(() => setSaveMessage(''), 3000);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 theme-surface">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold theme-text mb-2">Customize Your Experience</h1>
        <p className="theme-text-secondary">
          Personalize NeuroFlow to match your unique needs and preferences. All changes are saved automatically.
        </p>
        
        {saveMessage && (
          <div className="mt-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded-lg flex items-center gap-2">
            <Check className="w-4 h-4" />
            {saveMessage}
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 mb-6 border-b border-gray-200 pb-4">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-blue-500 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {/* Quick Presets */}
        {activeTab === 'presets' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold theme-text flex items-center gap-2">
              <Zap className="w-6 h-6" />
              Research-Backed Presets
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  name: 'Autism-Friendly',
                  icon: Brain,
                  description: 'Reduced sensory load, calming colors, increased spacing for comfort',
                  features: ['Muted colors', 'Extra spacing', 'No flashing', 'Soft transitions'],
                  color: 'bg-blue-100 hover:bg-blue-200',
                },
                {
                  name: 'ADHD Focus',
                  icon: Focus,
                  description: 'High contrast, clear focus indicators, minimal distractions',
                  features: ['High contrast', 'Strong focus rings', 'Clear structure', 'Bright accents'],
                  color: 'bg-purple-100 hover:bg-purple-200',
                },
                {
                  name: 'Dyslexia Support',
                  icon: Type,
                  description: 'Dyslexia-friendly fonts, spacing, and color combinations',
                  features: ['OpenDyslexic font', 'Wide spacing', 'Cream background', 'Clear text'],
                  color: 'bg-yellow-100 hover:bg-yellow-200',
                },
                {
                  name: 'Sensory Minimal',
                  icon: Heart,
                  description: 'Ultra-minimal design with neutral colors and reduced stimulation',
                  features: ['Neutral tones', 'No animations', 'Simple layout', 'Soft colors'],
                  color: 'bg-gray-100 hover:bg-gray-200',
                },
              ].map((preset) => {
                const Icon = preset.icon;
                return (
                  <div
                    key={preset.name}
                    className={`p-6 rounded-xl border-2 border-gray-200 ${preset.color} transition-all duration-200 cursor-pointer`}
                    onClick={() => applyPreset(preset.name)}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <Icon className="w-8 h-8 text-gray-700" />
                      <h3 className="text-xl font-semibold text-gray-800">{preset.name}</h3>
                    </div>
                    <p className="text-gray-600 mb-4">{preset.description}</p>
                    <div className="flex flex-wrap gap-2">
                      {preset.features.map((feature) => (
                        <span
                          key={feature}
                          className="px-2 py-1 bg-white rounded-full text-sm text-gray-600 border"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="flex gap-4 pt-4">
              <button
                onClick={() => themeStore.resetToDefaults()}
                className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                Reset to Default
              </button>
              <button
                onClick={handleExport}
                className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              >
                <Download className="w-4 h-4" />
                Export Settings
              </button>
              <button
                onClick={() => setShowImportDialog(true)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                <Upload className="w-4 h-4" />
                Import Settings
              </button>
            </div>
          </div>
        )}

        {/* Colors Tab */}
        {activeTab === 'colors' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold theme-text flex items-center gap-2">
              <Palette className="w-6 h-6" />
              Color Palettes
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(COLOR_PALETTES).map(([key, palette]) => (
                <div
                  key={key}
                  className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                    themeStore.colorPalette === key
                      ? 'border-blue-500 shadow-lg'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => themeStore.setColorPalette(key as keyof typeof COLOR_PALETTES)}
                >
                  <h3 className="font-semibold text-gray-800 mb-2">{palette.name}</h3>
                  <p className="text-sm text-gray-600 mb-3">{palette.description}</p>
                  <div className="flex gap-2">
                    <div className="w-6 h-6 rounded-full" style={{ backgroundColor: palette.primary }}></div>
                    <div className="w-6 h-6 rounded-full" style={{ backgroundColor: palette.secondary }}></div>
                    <div className="w-6 h-6 rounded-full" style={{ backgroundColor: palette.accent }}></div>
                    <div className="w-6 h-6 rounded-full border" style={{ backgroundColor: palette.background }}></div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={themeStore.darkMode}
                  onChange={(e) => themeStore.setDarkMode(e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-gray-700">Dark Mode</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={themeStore.highContrast}
                  onChange={(e) => themeStore.setHighContrast(e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-gray-700">High Contrast</span>
              </label>
            </div>
          </div>
        )}

        {/* Typography Tab */}
        {activeTab === 'typography' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold theme-text flex items-center gap-2">
              <Type className="w-6 h-6" />
              Typography
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Font Family
                </label>
                <select
                  value={themeStore.fontFamily}
                  onChange={(e) => themeStore.setFontFamily(e.target.value as any)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="inter">Inter (Default)</option>
                  <option value="poppins">Poppins (Friendly)</option>
                  <option value="opendyslexic">OpenDyslexic (Dyslexia-friendly)</option>
                  <option value="comic-sans">Comic Sans (Casual)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Font Size: {themeStore.fontSize}rem
                </label>
                <input
                  type="range"
                  min="0.8"
                  max="2"
                  step="0.1"
                  value={themeStore.fontSize}
                  onChange={(e) => themeStore.setFontSize(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Line Height: {themeStore.lineHeight}
                </label>
                <input
                  type="range"
                  min="1"
                  max="2.5"
                  step="0.1"
                  value={themeStore.lineHeight}
                  onChange={(e) => themeStore.setLineHeight(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Letter Spacing: {themeStore.letterSpacing}px
                </label>
                <input
                  type="range"
                  min="-1"
                  max="3"
                  step="0.1"
                  value={themeStore.letterSpacing}
                  onChange={(e) => themeStore.setLetterSpacing(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-800 mb-2">Preview</h3>
              <p className="text-lg" style={{
                fontFamily: `var(--font-family)`,
                fontSize: `${themeStore.fontSize}rem`,
                lineHeight: themeStore.lineHeight,
                letterSpacing: `${themeStore.letterSpacing}px`
              }}>
                This is how your text will appear throughout the application. 
                Adjusting these settings can significantly improve readability and comfort.
              </p>
            </div>
          </div>
        )}

        {/* Accessibility Tab */}
        {activeTab === 'accessibility' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold theme-text flex items-center gap-2">
              <Accessibility className="w-6 h-6" />
              Accessibility
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                  <div>
                    <div className="font-medium text-gray-800">Reduced Motion</div>
                    <div className="text-sm text-gray-600">Minimize animations and transitions</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={themeStore.reducedMotion}
                    onChange={(e) => themeStore.setReducedMotion(e.target.checked)}
                    className="w-5 h-5"
                  />
                </label>

                <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                  <div>
                    <div className="font-medium text-gray-800">Focus Ring Visible</div>
                    <div className="text-sm text-gray-600">Show focus indicators for keyboard navigation</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={themeStore.focusRingVisible}
                    onChange={(e) => themeStore.setFocusRingVisible(e.target.checked)}
                    className="w-5 h-5"
                  />
                </label>

                <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                  <div>
                    <div className="font-medium text-gray-800">Disable Flashing</div>
                    <div className="text-sm text-gray-600">Prevent flashing content that may trigger seizures</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={themeStore.flashingDisabled}
                    onChange={(e) => themeStore.setFlashingDisabled(e.target.checked)}
                    className="w-5 h-5"
                  />
                </label>
              </div>

              <div className="space-y-4">
                <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                  <div>
                    <div className="font-medium text-gray-800">Disable Autoplay</div>
                    <div className="text-sm text-gray-600">Prevent videos and audio from autoplaying</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={themeStore.autoplayDisabled}
                    onChange={(e) => themeStore.setAutoplayDisabled(e.target.checked)}
                    className="w-5 h-5"
                  />
                </label>

                <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
                  <div className="flex items-center gap-2">
                    {themeStore.soundEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
                    <div>
                      <div className="font-medium text-gray-800">Sound Feedback</div>
                      <div className="text-sm text-gray-600">Audio cues for interactions</div>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={themeStore.soundEnabled}
                    onChange={(e) => themeStore.setSoundEnabled(e.target.checked)}
                    className="w-5 h-5"
                  />
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Visual Tab */}
        {activeTab === 'visual' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold theme-text flex items-center gap-2">
              <Eye className="w-6 h-6" />
              Visual Layout
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Corner Radius: {themeStore.cornerRadius}px
                </label>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="1"
                  value={themeStore.cornerRadius}
                  onChange={(e) => themeStore.setCornerRadius(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="mt-2 flex justify-between text-sm text-gray-500">
                  <span>Sharp</span>
                  <span>Rounded</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Spacing: {themeStore.spacing}x
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={themeStore.spacing}
                  onChange={(e) => themeStore.setSpacing(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="mt-2 flex justify-between text-sm text-gray-500">
                  <span>Compact</span>
                  <span>Spacious</span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg" style={{ borderRadius: `${themeStore.cornerRadius}px` }}>
              <h3 className="font-semibold text-gray-800 mb-2">Preview</h3>
              <div 
                className="p-4 bg-white shadow-sm border"
                style={{ 
                  borderRadius: `${themeStore.cornerRadius}px`,
                  margin: `${themeStore.spacing * 8}px 0`
                }}
              >
                <p>This card shows how your corner radius and spacing settings affect the interface.</p>
              </div>
            </div>
          </div>
        )}

        {/* Personal Tab */}
        {activeTab === 'personal' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold theme-text flex items-center gap-2">
              <Heart className="w-6 h-6" />
              Personal Preferences
            </h2>
            
            <div className="space-y-4">
              <div className="p-6 bg-blue-50 rounded-lg border border-blue-200">
                <h3 className="font-semibold text-blue-800 mb-2">Your Comfort Matters</h3>
                <p className="text-blue-700">
                  These settings are automatically saved and will be remembered across all your sessions. 
                  You can export your settings to use them on other devices or share with your care team.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-800 mb-2">Current Configuration</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Color Palette: {COLOR_PALETTES[themeStore.colorPalette].name}</li>
                    <li>• Font Size: {themeStore.fontSize}rem</li>
                    <li>• Font Family: {themeStore.fontFamily}</li>
                    <li>• Dark Mode: {themeStore.darkMode ? 'Enabled' : 'Disabled'}</li>
                    <li>• Reduced Motion: {themeStore.reducedMotion ? 'Enabled' : 'Disabled'}</li>
                  </ul>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-800 mb-2">Quick Actions</h4>
                  <div className="space-y-2">
                    <button
                      onClick={handleExport}
                      className="w-full text-left px-3 py-2 bg-white rounded border hover:bg-gray-50 transition-colors flex items-center gap-2"
                    >
                      <Save className="w-4 h-4" />
                      Export Settings
                    </button>
                    <button
                      onClick={() => themeStore.resetToDefaults()}
                      className="w-full text-left px-3 py-2 bg-white rounded border hover:bg-gray-50 transition-colors flex items-center gap-2"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Reset to Defaults
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Import Dialog */}
      {showImportDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Import Settings</h3>
            <textarea
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
              placeholder="Paste your exported settings here..."
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-none"
            />
            <div className="flex gap-3 mt-4">
              <button
                onClick={handleImport}
                className="flex-1 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
              >
                <Upload className="w-4 h-4" />
                Import
              </button>
              <button
                onClick={() => {
                  setShowImportDialog(false);
                  setImportText('');
                }}
                className="flex-1 bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors flex items-center justify-center gap-2"
              >
                <X className="w-4 h-4" />
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FunctionalSettings;
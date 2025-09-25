import React, { useState } from 'react';
import { useTaskStore } from '../../hooks/useTaskStore';
import { CreateTaskRequest, TaskPriority, TaskComplexity, EnergyLevel, TaskCategory } from '../../types/task';
import { FiX, FiPlus, FiMinus, FiCalendar, FiTag, FiAlertCircle, FiInfo, FiZap, FiClock, FiTarget } from 'react-icons/fi';

interface TaskCreateFormProps {
  isOpen: boolean;
  onClose: () => void;
  parentTaskId?: string;
}

const TaskCreateForm: React.FC<TaskCreateFormProps> = ({ isOpen, onClose, parentTaskId }) => {
  const { createTask, isLoading } = useTaskStore();
  
  const [formData, setFormData] = useState<Omit<CreateTaskRequest, 'id'>>({
    title: '',
    description: '',
    priority: TaskPriority.MEDIUM,
    complexity: TaskComplexity.MODERATE,
    estimatedMinutes: 30,
    energyLevel: EnergyLevel.MEDIUM,
    category: TaskCategory.WORK,
    tags: [],
    dueDate: undefined,
    parentTaskId,
  });

  const [currentTag, setCurrentTag] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showComplexityTips, setShowComplexityTips] = useState(false);
  const [showEnergyTips, setShowEnergyTips] = useState(false);

  // Complexity assessment tips
  const complexityTips = {
    [TaskComplexity.SIMPLE]: {
      description: "Quick, single-step tasks that take minimal mental effort",
      examples: ["Reply to an email", "Make a quick phone call", "File a document"],
      timeRange: "5-15 minutes",
      energyNeeded: "Low"
    },
    [TaskComplexity.MODERATE]: {
      description: "Multi-step tasks that require some planning and focus",
      examples: ["Write a report section", "Research a topic", "Organize files"],
      timeRange: "15-60 minutes", 
      energyNeeded: "Medium"
    },
    [TaskComplexity.COMPLEX]: {
      description: "Tasks requiring deep focus, planning, and sustained attention",
      examples: ["Design a system", "Write a proposal", "Learn new skills"],
      timeRange: "1-4 hours",
      energyNeeded: "High"
    },
    [TaskComplexity.OVERWHELMING]: {
      description: "Large tasks that should be broken down into smaller chunks",
      examples: ["Complete a project", "Redesign workflow", "Major presentation"],
      timeRange: "4+ hours",
      energyNeeded: "Very High"
    }
  };

  const energyTips = {
    [EnergyLevel.LOW]: {
      description: "Tasks you can do when tired, distracted, or low on mental energy",
      examples: ["Organize emails", "Simple data entry", "Routine administrative tasks"],
      bestTime: "End of day or during energy dips"
    },
    [EnergyLevel.MEDIUM]: {
      description: "Tasks requiring moderate focus and mental engagement",
      examples: ["Writing drafts", "Planning meetings", "Review and edit work"],
      bestTime: "Mid-morning or early afternoon"
    },
    [EnergyLevel.HIGH]: {
      description: "Tasks requiring peak mental performance and deep focus",
      examples: ["Creative work", "Problem-solving", "Important decisions"],
      bestTime: "Your personal peak energy hours"
    }
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.title.trim()) {
      newErrors.title = 'Task title is required';
    }

    if (formData.estimatedMinutes < 5) {
      newErrors.estimatedMinutes = 'Estimated time should be at least 5 minutes';
    }

    if (formData.estimatedMinutes > 480) {
      newErrors.estimatedMinutes = 'Tasks over 8 hours should be broken down into smaller tasks';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    try {
      await createTask(formData);
      handleClose();
    } catch (error) {
      setErrors({ general: 'Failed to create task. Please try again.' });
    }
  };

  const handleClose = () => {
    setFormData({
      title: '',
      description: '',
      priority: TaskPriority.MEDIUM,
      complexity: TaskComplexity.MODERATE,
      estimatedMinutes: 30,
      energyLevel: EnergyLevel.MEDIUM,
      category: TaskCategory.WORK,
      tags: [],
      dueDate: undefined,
      parentTaskId,
    });
    setCurrentTag('');
    setErrors({});
    setShowComplexityTips(false);
    setShowEnergyTips(false);
    onClose();
  };

  const addTag = () => {
    const tags = formData.tags || [];
    if (currentTag.trim() && !tags.includes(currentTag.trim())) {
      setFormData(prev => ({
        ...prev,
        tags: [...(prev.tags || []), currentTag.trim()]
      }));
      setCurrentTag('');
    }
  };

  const removeTag = (tagToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      tags: (prev.tags || []).filter(tag => tag !== tagToRemove)
    }));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && currentTag.trim()) {
      e.preventDefault();
      addTag();
    }
  };

  // Auto-suggest energy level based on complexity
  const getSuggestedEnergyLevel = (complexity: TaskComplexity): EnergyLevel => {
    switch (complexity) {
      case TaskComplexity.SIMPLE:
        return EnergyLevel.LOW;
      case TaskComplexity.MODERATE:
        return EnergyLevel.MEDIUM;
      case TaskComplexity.COMPLEX:
      case TaskComplexity.OVERWHELMING:
        return EnergyLevel.HIGH;
      default:
        return EnergyLevel.MEDIUM;
    }
  };

  // Auto-adjust estimated time based on complexity
  const getSuggestedTime = (complexity: TaskComplexity): number => {
    switch (complexity) {
      case TaskComplexity.SIMPLE:
        return 15;
      case TaskComplexity.MODERATE:
        return 30;
      case TaskComplexity.COMPLEX:
        return 90;
      case TaskComplexity.OVERWHELMING:
        return 240;
      default:
        return 30;
    }
  };

  const handleComplexityChange = (complexity: TaskComplexity) => {
    setFormData(prev => ({
      ...prev,
      complexity,
      energyLevel: getSuggestedEnergyLevel(complexity),
      estimatedMinutes: getSuggestedTime(complexity)
    }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
                {parentTaskId ? 'Create Subtask' : 'Create New Task'}
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                Break down your work into manageable, neurodivergent-friendly tasks
              </p>
            </div>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600 transition-colors p-2 rounded-full hover:bg-gray-100"
            >
              <FiX className="w-6 h-6" />
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {errors.general && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-2">
              <FiAlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-red-700">{errors.general}</span>
            </div>
          )}

          {/* Task Title */}
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-2">
              Task Title *
            </label>
            <input
              type="text"
              id="title"
              value={formData.title}
              onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
              className={`w-full border rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                errors.title ? 'border-red-300' : 'border-gray-300'
              }`}
              placeholder="What do you need to accomplish?"
            />
            {errors.title && <p className="text-red-600 text-sm mt-1">{errors.title}</p>}
          </div>

          {/* Task Description */}
          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
              Description (Optional)
            </label>
            <textarea
              id="description"
              rows={3}
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder="Add details, context, or notes about this task..."
            />
          </div>

          {/* Priority and Category Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label htmlFor="priority" className="block text-sm font-medium text-gray-700 mb-2">
                Priority
              </label>
              <select
                id="priority"
                value={formData.priority}
                onChange={(e) => setFormData(prev => ({ ...prev, priority: e.target.value as TaskPriority }))}
                className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500"
              >
                <option value={TaskPriority.LOW}>🟢 Low - When convenient</option>
                <option value={TaskPriority.MEDIUM}>🟡 Medium - Should be done</option>
                <option value={TaskPriority.HIGH}>🟠 High - Important</option>
                <option value={TaskPriority.URGENT}>🔴 Urgent - Do ASAP</option>
              </select>
            </div>

            <div>
              <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
                Category
              </label>
              <select
                id="category"
                value={formData.category}
                onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value as TaskCategory }))}
                className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500"
              >
                <option value={TaskCategory.WORK}>💼 Work</option>
                <option value={TaskCategory.PERSONAL}>🏠 Personal</option>
                <option value={TaskCategory.HEALTH}>🏃 Health & Wellness</option>
                <option value={TaskCategory.LEARNING}>📚 Learning</option>
                <option value={TaskCategory.SOCIAL}>👥 Social</option>
                <option value={TaskCategory.ADMINISTRATIVE}>📋 Administrative</option>
              </select>
            </div>
          </div>

          {/* Complexity Assessment */}
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Task Complexity
              </label>
              <button
                type="button"
                onClick={() => setShowComplexityTips(!showComplexityTips)}
                className="text-blue-500 hover:text-blue-700 transition-colors"
              >
                <FiInfo className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
              {Object.values(TaskComplexity).map((complexity) => (
                <button
                  key={complexity}
                  type="button"
                  onClick={() => handleComplexityChange(complexity)}
                  className={`p-3 rounded-lg border-2 text-center transition-colors ${
                    formData.complexity === complexity
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300 text-gray-700'
                  }`}
                >
                  <div className="font-medium capitalize mb-1">{complexity}</div>
                  <div className="text-xs">
                    {complexity === TaskComplexity.SIMPLE && '5-15min'}
                    {complexity === TaskComplexity.MODERATE && '15-60min'}
                    {complexity === TaskComplexity.COMPLEX && '1-4hrs'}
                    {complexity === TaskComplexity.OVERWHELMING && '4+ hrs'}
                  </div>
                </button>
              ))}
            </div>

            {showComplexityTips && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <FiTarget className="w-5 h-5 text-blue-600" />
                  <h4 className="font-medium text-blue-900 capitalize">
                    {formData.complexity} Tasks
                  </h4>
                </div>
                <p className="text-blue-800 text-sm mb-2">
                  {complexityTips[formData.complexity].description}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-blue-900">Examples:</span>
                    <ul className="text-blue-800 mt-1 space-y-1">
                      {complexityTips[formData.complexity].examples.map((example, index) => (
                        <li key={index}>• {example}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <span className="font-medium text-blue-900">Time Range:</span>
                    <p className="text-blue-800 mt-1">{complexityTips[formData.complexity].timeRange}</p>
                  </div>
                  <div>
                    <span className="font-medium text-blue-900">Energy Needed:</span>
                    <p className="text-blue-800 mt-1">{complexityTips[formData.complexity].energyNeeded}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Energy Level and Time Estimation Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <label className="block text-sm font-medium text-gray-700">
                  Energy Level Required
                </label>
                <button
                  type="button"
                  onClick={() => setShowEnergyTips(!showEnergyTips)}
                  className="text-blue-500 hover:text-blue-700 transition-colors"
                >
                  <FiInfo className="w-4 h-4" />
                </button>
              </div>

              <div className="grid grid-cols-3 gap-2">
                {Object.values(EnergyLevel).map((energy) => (
                  <button
                    key={energy}
                    type="button"
                    onClick={() => setFormData(prev => ({ ...prev, energyLevel: energy }))}
                    className={`p-3 rounded-lg border-2 text-center transition-colors ${
                      formData.energyLevel === energy
                        ? 'border-green-500 bg-green-50 text-green-700'
                        : 'border-gray-200 hover:border-gray-300 text-gray-700'
                    }`}
                  >
                    <FiZap className="w-4 h-4 mx-auto mb-1" />
                    <div className="font-medium text-xs capitalize">{energy}</div>
                  </button>
                ))}
              </div>

              {showEnergyTips && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 mt-2">
                  <div className="flex items-center space-x-2 mb-1">
                    <FiZap className="w-4 h-4 text-green-600" />
                    <h5 className="font-medium text-green-900 text-sm capitalize">
                      {formData.energyLevel} Energy Tasks
                    </h5>
                  </div>
                  <p className="text-green-800 text-xs mb-2">
                    {energyTips[formData.energyLevel].description}
                  </p>
                  <p className="text-green-700 text-xs">
                    <span className="font-medium">Best time:</span> {energyTips[formData.energyLevel].bestTime}
                  </p>
                </div>
              )}
            </div>

            <div>
              <label htmlFor="estimatedMinutes" className="block text-sm font-medium text-gray-700 mb-2">
                Estimated Time (minutes)
              </label>
              <div className="relative">
                <FiClock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="number"
                  id="estimatedMinutes"
                  min="5"
                  max="480"
                  value={formData.estimatedMinutes}
                  onChange={(e) => setFormData(prev => ({ ...prev, estimatedMinutes: parseInt(e.target.value) || 0 }))}
                  className={`w-full border rounded-lg pl-10 pr-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                    errors.estimatedMinutes ? 'border-red-300' : 'border-gray-300'
                  }`}
                />
              </div>
              {errors.estimatedMinutes && (
                <p className="text-red-600 text-sm mt-1">{errors.estimatedMinutes}</p>
              )}
              <div className="text-xs text-gray-500 mt-1">
                ≈ {Math.round(formData.estimatedMinutes / 60 * 10) / 10} hours
              </div>
            </div>
          </div>

          {/* Due Date */}
          <div>
            <label htmlFor="dueDate" className="block text-sm font-medium text-gray-700 mb-2">
              Due Date (Optional)
            </label>
            <div className="relative">
              <FiCalendar className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="date"
                id="dueDate"
                value={formData.dueDate ? new Date(formData.dueDate).toISOString().split('T')[0] : ''}
                onChange={(e) => setFormData(prev => ({
                  ...prev,
                  dueDate: e.target.value ? new Date(e.target.value) : undefined
                }))}
                className="w-full border border-gray-300 rounded-lg pl-10 pr-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tags (Optional)
            </label>
            <div className="flex items-center space-x-2 mb-3">
              <div className="relative flex-1">
                <FiTag className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  value={currentTag}
                  onChange={(e) => setCurrentTag(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="w-full border border-gray-300 rounded-lg pl-10 pr-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Add a tag..."
                />
              </div>
              <button
                type="button"
                onClick={addTag}
                disabled={!currentTag.trim()}
                className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                <FiPlus className="w-5 h-5" />
              </button>
            </div>

            {(formData.tags || []).length > 0 && (
              <div className="flex flex-wrap gap-2">
                {(formData.tags || []).map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center space-x-1 px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm"
                  >
                    <span>#{tag}</span>
                    <button
                      type="button"
                      onClick={() => removeTag(tag)}
                      className="text-gray-500 hover:text-gray-700 transition-colors"
                    >
                      <FiMinus className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Submit Buttons */}
          <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-200">
            <button
              type="button"
              onClick={handleClose}
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
            >
              {isLoading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>}
              <span>{parentTaskId ? 'Create Subtask' : 'Create Task'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default TaskCreateForm;
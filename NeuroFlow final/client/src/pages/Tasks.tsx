import React, { useEffect, useState } from 'react';
import { useTaskStore } from '../hooks/useTaskStore';
import { TaskStatus, TaskPriority, TaskComplexity, EnergyLevel, TaskCategory } from '../types/task';
import { FiPlus, FiList, FiGrid, FiCalendar, FiFilter, FiSearch, FiSettings, FiTrendingUp, FiTarget, FiZap, FiClock, FiAward } from 'react-icons/fi';
import TaskCreateForm from '../components/tasks/TaskCreateForm';

const Tasks: React.FC = () => {
  const {
    tasks,
    userStats,
    filters,
    sortOptions,
    viewMode,
    showCompleted,
    groupBy,
    isLoading,
    error,
    getFilteredTasks,
    getTasksByStatus,
    getOverdueTasks,
    getTasksForToday,
    setFilters,
    setSortOptions,
    setViewMode,
    setGroupBy,
    clearFilters,
    selectTask,
    initializeWithSampleData,
  } = useTaskStore();

  const [searchQuery, setSearchQuery] = useState(filters.searchQuery || '');
  const [showFilters, setShowFilters] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);

  // Initialize sample data on first load
  useEffect(() => {
    if (tasks.length === 0) {
      initializeWithSampleData();
    }
  }, [tasks.length, initializeWithSampleData]);

  // Update search filter when searchQuery changes
  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      setFilters({ searchQuery });
    }, 300);

    return () => clearTimeout(debounceTimer);
  }, [searchQuery, setFilters]);

  const filteredTasks = getFilteredTasks();
  const todoTasks = getTasksByStatus(TaskStatus.TODO);
  const inProgressTasks = getTasksByStatus(TaskStatus.IN_PROGRESS);
  const completedTasks = getTasksByStatus(TaskStatus.COMPLETED);
  const overdueTasks = getOverdueTasks();
  const todayTasks = getTasksForToday();

  // Quick stats for dashboard
  const quickStats = {
    total: tasks.length,
    completed: completedTasks.length,
    overdue: overdueTasks.length,
    today: todayTasks.length,
    completionRate: userStats.completionRate,
    currentStreak: userStats.currentStreak,
    level: userStats.level,
    totalPoints: userStats.totalPoints,
  };

  const handleViewModeChange = (mode: 'list' | 'board' | 'calendar') => {
    setViewMode(mode);
  };

  const handleSortChange = (field: string) => {
    const newDirection = sortOptions.field === field && sortOptions.direction === 'desc' ? 'asc' : 'desc';
    setSortOptions({ field: field as any, direction: newDirection });
  };

  const renderQuickStats = () => (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center space-x-3">
        <div className="bg-blue-100 p-2 rounded-full">
          <FiTarget className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <div className="text-2xl font-bold text-blue-900">{quickStats.total}</div>
          <div className="text-sm text-blue-600">Total Tasks</div>
        </div>
      </div>
      
      <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center space-x-3">
        <div className="bg-green-100 p-2 rounded-full">
          <FiZap className="w-5 h-5 text-green-600" />
        </div>
        <div>
          <div className="text-2xl font-bold text-green-900">{quickStats.completed}</div>
          <div className="text-sm text-green-600">Completed</div>
        </div>
      </div>
      
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
        <div className="bg-red-100 p-2 rounded-full">
          <FiTrendingUp className="w-5 h-5 text-red-600" />
        </div>
        <div>
          <div className="text-2xl font-bold text-red-900">{quickStats.overdue}</div>
          <div className="text-sm text-red-600">Overdue</div>
        </div>
      </div>
      
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 flex items-center space-x-3">
        <div className="bg-purple-100 p-2 rounded-full">
          <FiClock className="w-5 h-5 text-purple-600" />
        </div>
        <div>
          <div className="text-2xl font-bold text-purple-900">{quickStats.today}</div>
          <div className="text-sm text-purple-600">Due Today</div>
        </div>
      </div>
      
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-center space-x-3">
        <div className="bg-yellow-100 p-2 rounded-full">
          <FiAward className="w-5 h-5 text-yellow-600" />
        </div>
        <div>
          <div className="text-2xl font-bold text-yellow-900">{quickStats.currentStreak}</div>
          <div className="text-sm text-yellow-600">Day Streak</div>
        </div>
      </div>
      
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 flex items-center space-x-3">
        <div className="bg-indigo-100 p-2 rounded-full">
          <FiSettings className="w-5 h-5 text-indigo-600" />
        </div>
        <div>
          <div className="text-2xl font-bold text-indigo-900">Lv.{quickStats.level}</div>
          <div className="text-sm text-indigo-600">{quickStats.totalPoints} pts</div>
        </div>
      </div>
    </div>
  );

  const renderToolbar = () => (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm mb-6">
      <div className="p-4">
        {/* Top row - Main actions and search */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0 mb-4">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowCreateForm(true)}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              <FiPlus className="w-5 h-5" />
              <span>Add Task</span>
            </button>
            
            <div className="relative">
              <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search tasks..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent w-64"
              />
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`px-3 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                showFilters 
                  ? 'bg-blue-100 text-blue-700 border border-blue-300' 
                  : 'bg-gray-100 text-gray-700 border border-gray-300 hover:bg-gray-200'
              }`}
            >
              <FiFilter className="w-4 h-4" />
              <span>Filters</span>
            </button>
            
            <div className="flex items-center bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => handleViewModeChange('list')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'list' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-600 hover:text-gray-900'
                }`}
                title="List View"
              >
                <FiList className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleViewModeChange('board')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'board' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-600 hover:text-gray-900'
                }`}
                title="Board View"
              >
                <FiGrid className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleViewModeChange('calendar')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'calendar' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-600 hover:text-gray-900'
                }`}
                title="Calendar View"
              >
                <FiCalendar className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Filters panel */}
        {showFilters && (
          <div className="border-t border-gray-200 pt-4">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                <select
                  multiple
                  value={filters.status || []}
                  onChange={(e) => {
                    const values = Array.from(e.target.selectedOptions).map(option => option.value as TaskStatus);
                    setFilters({ status: values });
                  }}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <option value={TaskStatus.TODO}>To Do</option>
                  <option value={TaskStatus.IN_PROGRESS}>In Progress</option>
                  <option value={TaskStatus.COMPLETED}>Completed</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
                <select
                  multiple
                  value={filters.priority || []}
                  onChange={(e) => {
                    const values = Array.from(e.target.selectedOptions).map(option => option.value as TaskPriority);
                    setFilters({ priority: values });
                  }}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <option value={TaskPriority.URGENT}>Urgent</option>
                  <option value={TaskPriority.HIGH}>High</option>
                  <option value={TaskPriority.MEDIUM}>Medium</option>
                  <option value={TaskPriority.LOW}>Low</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Complexity</label>
                <select
                  multiple
                  value={filters.complexity || []}
                  onChange={(e) => {
                    const values = Array.from(e.target.selectedOptions).map(option => option.value as TaskComplexity);
                    setFilters({ complexity: values });
                  }}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <option value={TaskComplexity.SIMPLE}>Simple</option>
                  <option value={TaskComplexity.MODERATE}>Moderate</option>
                  <option value={TaskComplexity.COMPLEX}>Complex</option>
                  <option value={TaskComplexity.OVERWHELMING}>Overwhelming</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Energy Level</label>
                <select
                  multiple
                  value={filters.energyLevel || []}
                  onChange={(e) => {
                    const values = Array.from(e.target.selectedOptions).map(option => option.value as EnergyLevel);
                    setFilters({ energyLevel: values });
                  }}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <option value={EnergyLevel.LOW}>Low</option>
                  <option value={EnergyLevel.MEDIUM}>Medium</option>
                  <option value={EnergyLevel.HIGH}>High</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                <select
                  multiple
                  value={filters.category || []}
                  onChange={(e) => {
                    const values = Array.from(e.target.selectedOptions).map(option => option.value as TaskCategory);
                    setFilters({ category: values });
                  }}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500"
                >
                  <option value={TaskCategory.WORK}>Work</option>
                  <option value={TaskCategory.PERSONAL}>Personal</option>
                  <option value={TaskCategory.HEALTH}>Health</option>
                  <option value={TaskCategory.LEARNING}>Learning</option>
                  <option value={TaskCategory.SOCIAL}>Social</option>
                  <option value={TaskCategory.ADMINISTRATIVE}>Administrative</option>
                </select>
              </div>

              <div className="flex items-end">
                <button
                  onClick={clearFilters}
                  className="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg transition-colors text-sm"
                >
                  Clear Filters
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderTaskList = () => (
    <div className="space-y-4">
      {filteredTasks.length === 0 ? (
        <div className="bg-white border border-gray-200 rounded-lg p-12 text-center">
          <FiTarget className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No tasks found</h3>
          <p className="text-gray-600 mb-6">
            {tasks.length === 0 
              ? "Get started by creating your first task!" 
              : "Try adjusting your filters or search query."
            }
          </p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 mx-auto transition-colors"
          >
            <FiPlus className="w-5 h-5" />
            <span>Create Task</span>
          </button>
        </div>
      ) : (
        filteredTasks.map((task) => (
          <div
            key={task.id}
            onClick={() => selectTask(task)}
            className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="text-lg font-medium text-gray-900">{task.title}</h3>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      task.priority === TaskPriority.URGENT ? 'bg-red-100 text-red-800' :
                      task.priority === TaskPriority.HIGH ? 'bg-orange-100 text-orange-800' :
                      task.priority === TaskPriority.MEDIUM ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {task.priority}
                    </span>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      task.status === TaskStatus.COMPLETED ? 'bg-green-100 text-green-800' :
                      task.status === TaskStatus.IN_PROGRESS ? 'bg-blue-100 text-blue-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {task.status.replace('_', ' ')}
                    </span>
                  </div>
                </div>
                {task.description && (
                  <p className="text-gray-600 mb-2">{task.description}</p>
                )}
                <div className="flex items-center space-x-4 text-sm text-gray-500">
                  <span>Complexity: {task.complexity}</span>
                  <span>Energy: {task.energyLevel}</span>
                  <span>Est: {task.estimatedMinutes}min</span>
                  {task.dueDate && (
                    <span>Due: {new Date(task.dueDate).toLocaleDateString()}</span>
                  )}
                </div>
                {task.tags.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {task.tags.map((tag, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-full"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  );

  const renderTaskBoard = () => {
    const columns = [
      { status: TaskStatus.TODO, title: 'To Do', color: 'bg-gray-50 border-gray-200' },
      { status: TaskStatus.IN_PROGRESS, title: 'In Progress', color: 'bg-blue-50 border-blue-200' },
      { status: TaskStatus.COMPLETED, title: 'Completed', color: 'bg-green-50 border-green-200' },
    ];

    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {columns.map((column) => {
          const columnTasks = filteredTasks.filter(task => task.status === column.status);
          
          return (
            <div key={column.status} className={`rounded-lg border p-4 ${column.color}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-gray-900">{column.title}</h3>
                <span className="bg-gray-200 text-gray-700 text-sm px-2 py-1 rounded-full">
                  {columnTasks.length}
                </span>
              </div>
              
              <div className="space-y-3">
                {columnTasks.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <p>No {column.title.toLowerCase()} tasks</p>
                  </div>
                ) : (
                  columnTasks.map((task) => (
                    <div
                      key={task.id}
                      onClick={() => selectTask(task)}
                      className="bg-white border border-gray-200 rounded-lg p-3 hover:shadow-sm transition-shadow cursor-pointer"
                    >
                      <h4 className="font-medium text-gray-900 mb-1">{task.title}</h4>
                      {task.description && (
                        <p className="text-sm text-gray-600 mb-2 line-clamp-2">{task.description}</p>
                      )}
                      <div className="flex items-center space-x-2 mb-2">
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                          task.priority === TaskPriority.URGENT ? 'bg-red-100 text-red-800' :
                          task.priority === TaskPriority.HIGH ? 'bg-orange-100 text-orange-800' :
                          task.priority === TaskPriority.MEDIUM ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {task.priority}
                        </span>
                        <span className="text-xs text-gray-500">{task.estimatedMinutes}min</span>
                      </div>
                      {task.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {task.tags.slice(0, 2).map((tag, index) => (
                            <span key={index} className="px-1 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                              #{tag}
                            </span>
                          ))}
                          {task.tags.length > 2 && (
                            <span className="px-1 py-0.5 text-xs text-gray-500">
                              +{task.tags.length - 2}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderTaskCalendar = () => {
    // Get tasks with due dates
    const tasksWithDates = filteredTasks.filter(task => task.dueDate);
    
    // Group tasks by date
    const tasksByDate = tasksWithDates.reduce((acc, task) => {
      if (task.dueDate) {
        const dateKey = new Date(task.dueDate).toDateString();
        if (!acc[dateKey]) {
          acc[dateKey] = [];
        }
        acc[dateKey].push(task);
      }
      return acc;
    }, {} as Record<string, typeof tasksWithDates>);

    // Get current month info
    const today = new Date();
    const currentMonth = today.getMonth();
    const currentYear = today.getFullYear();
    const firstDayOfMonth = new Date(currentYear, currentMonth, 1);
    const lastDayOfMonth = new Date(currentYear, currentMonth + 1, 0);
    const startDate = new Date(firstDayOfMonth);
    startDate.setDate(startDate.getDate() - firstDayOfMonth.getDay()); // Start from Sunday

    const days = [];
    const current = new Date(startDate);
    
    // Generate 6 weeks (42 days) for calendar view
    for (let i = 0; i < 42; i++) {
      days.push(new Date(current));
      current.setDate(current.getDate() + 1);
    }

    const monthNames = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];

    return (
      <div className="bg-white border border-gray-200 rounded-lg">
        {/* Calendar Header */}
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">
            {monthNames[currentMonth]} {currentYear}
          </h3>
        </div>

        {/* Days of week header */}
        <div className="grid grid-cols-7 border-b border-gray-200">
          {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
            <div key={day} className="p-2 text-center text-sm font-medium text-gray-500 bg-gray-50">
              {day}
            </div>
          ))}
        </div>

        {/* Calendar Grid */}
        <div className="grid grid-cols-7">
          {days.map((day, index) => {
            const isCurrentMonth = day.getMonth() === currentMonth;
            const isToday = day.toDateString() === today.toDateString();
            const dayTasks = tasksByDate[day.toDateString()] || [];
            const hasOverdue = dayTasks.some(task => 
              task.dueDate && new Date(task.dueDate) < today && task.status !== TaskStatus.COMPLETED
            );

            return (
              <div
                key={index}
                className={`min-h-[100px] border-r border-b border-gray-100 p-2 ${
                  !isCurrentMonth ? 'bg-gray-50 text-gray-400' : 'bg-white'
                } ${isToday ? 'bg-blue-50' : ''}`}
              >
                <div className={`text-sm font-medium mb-1 ${
                  isToday ? 'bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center' : ''
                }`}>
                  {day.getDate()}
                </div>
                
                <div className="space-y-1">
                  {dayTasks.slice(0, 3).map((task) => (
                    <div
                      key={task.id}
                      onClick={() => selectTask(task)}
                      className={`text-xs p-1 rounded cursor-pointer truncate ${
                        task.status === TaskStatus.COMPLETED ? 'bg-green-100 text-green-800' :
                        hasOverdue && task.dueDate && new Date(task.dueDate) < today ? 'bg-red-100 text-red-800' :
                        task.priority === TaskPriority.URGENT ? 'bg-red-100 text-red-800' :
                        task.priority === TaskPriority.HIGH ? 'bg-orange-100 text-orange-800' :
                        'bg-gray-100 text-gray-700'
                      }`}
                      title={task.title}
                    >
                      {task.title}
                    </div>
                  ))}
                  {dayTasks.length > 3 && (
                    <div className="text-xs text-gray-500 text-center">
                      +{dayTasks.length - 3} more
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Tasks without dates */}
        {filteredTasks.filter(task => !task.dueDate).length > 0 && (
          <div className="p-4 border-t border-gray-200">
            <h4 className="font-medium text-gray-900 mb-3">Tasks without due dates</h4>
            <div className="space-y-2">
              {filteredTasks.filter(task => !task.dueDate).map((task) => (
                <div
                  key={task.id}
                  onClick={() => selectTask(task)}
                  className="flex items-center space-x-2 p-2 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100"
                >
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    task.priority === TaskPriority.URGENT ? 'bg-red-100 text-red-800' :
                    task.priority === TaskPriority.HIGH ? 'bg-orange-100 text-orange-800' :
                    task.priority === TaskPriority.MEDIUM ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {task.priority}
                  </span>
                  <span className="text-sm text-gray-900">{task.title}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCurrentView = () => {
    switch (viewMode) {
      case 'board':
        return renderTaskBoard();
      case 'calendar':
        return renderTaskCalendar();
      case 'list':
      default:
        return renderTaskList();
    }
  };

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-red-900 mb-2">Error</h2>
          <p className="text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-gentle-fade-in">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Tasks</h1>
        <p className="text-lg text-gray-600">
          Organize your work with neurodivergent-friendly task management
        </p>
      </div>

      {/* Quick Stats */}
      {renderQuickStats()}

      {/* Toolbar */}
      {renderToolbar()}

      {/* Main Content */}
      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
          ) : (
            renderCurrentView()
          )}
        </div>
      </div>

      {/* Task Creation Form */}
      <TaskCreateForm 
        isOpen={showCreateForm} 
        onClose={() => setShowCreateForm(false)} 
      />
    </div>
  );
};

export default Tasks;
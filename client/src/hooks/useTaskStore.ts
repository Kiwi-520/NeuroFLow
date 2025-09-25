import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { 
  Task, 
  CreateTaskRequest, 
  UpdateTaskRequest,
  TaskFilters,
  TaskSortOptions,
  TaskViewState,
  TaskStatus,
  TaskPriority,
  TaskComplexity,
  EnergyLevel,
  TaskCategory,
  UserTaskStats
} from '../types/task';

interface TaskStore extends TaskViewState {
  // Task data
  tasks: Task[];
  
  // User stats and progress
  userStats: UserTaskStats;
  
  // Loading states
  isLoading: boolean;
  error: string | null;
  
  // CRUD Operations
  createTask: (request: CreateTaskRequest) => Promise<Task>;
  updateTask: (request: UpdateTaskRequest) => Promise<Task>;
  deleteTask: (taskId: string) => Promise<void>;
  completeTask: (taskId: string, focusScore?: number, actualMinutes?: number) => Promise<void>;
  
  // Task management
  duplicateTask: (taskId: string) => Promise<Task>;
  addSubtask: (parentId: string, subtaskData: CreateTaskRequest) => Promise<Task>;
  moveTask: (taskId: string, newStatus: TaskStatus) => Promise<void>;
  
  // Filtering and sorting
  setFilters: (filters: Partial<TaskFilters>) => void;
  setSortOptions: (sortOptions: TaskSortOptions) => void;
  setViewMode: (mode: 'list' | 'board' | 'calendar') => void;
  setGroupBy: (groupBy: 'priority' | 'category' | 'dueDate' | 'none') => void;
  clearFilters: () => void;
  
  // Computed getters
  getFilteredTasks: () => Task[];
  getTasksByStatus: (status: TaskStatus) => Task[];
  getOverdueTasks: () => Task[];
  getTasksForToday: () => Task[];
  getSubtasks: (parentId: string) => Task[];
  
  // Stats and analytics
  updateUserStats: () => void;
  getCompletionStreak: () => number;
  
  // Utility functions
  selectTask: (task: Task | undefined) => void;
  searchTasks: (query: string) => Task[];
  
  // Reset and initialization
  resetStore: () => void;
  initializeWithSampleData: () => void;
}

// Default state
const defaultFilters: TaskFilters = {
  searchQuery: '',
};

const defaultSortOptions: TaskSortOptions = {
  field: 'priority',
  direction: 'desc',
};

const defaultStats: UserTaskStats = {
  totalCompleted: 0,
  currentStreak: 0,
  longestStreak: 0,
  totalPoints: 0,
  level: 1,
  completionRate: 0,
  averageFocusScore: 0,
};

// Helper functions
const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

const calculatePoints = (task: Task): number => {
  let points = 10; // Base points
  
  // Priority multiplier
  switch (task.priority) {
    case TaskPriority.URGENT: points *= 3; break;
    case TaskPriority.HIGH: points *= 2; break;
    case TaskPriority.MEDIUM: points *= 1.5; break;
    case TaskPriority.LOW: points *= 1; break;
  }
  
  // Complexity multiplier
  switch (task.complexity) {
    case TaskComplexity.OVERWHELMING: points *= 4; break;
    case TaskComplexity.COMPLEX: points *= 3; break;
    case TaskComplexity.MODERATE: points *= 2; break;
    case TaskComplexity.SIMPLE: points *= 1; break;
  }
  
  return Math.round(points);
};

const sortTasks = (tasks: Task[], sortOptions: TaskSortOptions): Task[] => {
  return [...tasks].sort((a, b) => {
    let aValue: any;
    let bValue: any;
    
    switch (sortOptions.field) {
      case 'priority':
        const priorityOrder = { 'urgent': 4, 'high': 3, 'medium': 2, 'low': 1 };
        aValue = priorityOrder[a.priority];
        bValue = priorityOrder[b.priority];
        break;
      case 'dueDate':
        aValue = a.dueDate ? new Date(a.dueDate).getTime() : Infinity;
        bValue = b.dueDate ? new Date(b.dueDate).getTime() : Infinity;
        break;
      case 'createdAt':
        aValue = new Date(a.createdAt).getTime();
        bValue = new Date(b.createdAt).getTime();
        break;
      case 'complexity':
        const complexityOrder = { 'overwhelming': 4, 'complex': 3, 'moderate': 2, 'simple': 1 };
        aValue = complexityOrder[a.complexity];
        bValue = complexityOrder[b.complexity];
        break;
      case 'estimatedMinutes':
        aValue = a.estimatedMinutes;
        bValue = b.estimatedMinutes;
        break;
      default:
        return 0;
    }
    
    if (sortOptions.direction === 'asc') {
      return aValue - bValue;
    } else {
      return bValue - aValue;
    }
  });
};

const filterTasks = (tasks: Task[], filters: TaskFilters): Task[] => {
  return tasks.filter(task => {
    // Status filter
    if (filters.status && filters.status.length > 0 && !filters.status.includes(task.status)) {
      return false;
    }
    
    // Priority filter
    if (filters.priority && filters.priority.length > 0 && !filters.priority.includes(task.priority)) {
      return false;
    }
    
    // Category filter
    if (filters.category && filters.category.length > 0 && !filters.category.includes(task.category)) {
      return false;
    }
    
    // Complexity filter
    if (filters.complexity && filters.complexity.length > 0 && !filters.complexity.includes(task.complexity)) {
      return false;
    }
    
    // Energy level filter
    if (filters.energyLevel && filters.energyLevel.length > 0 && !filters.energyLevel.includes(task.energyLevel)) {
      return false;
    }
    
    // Due within filter
    if (filters.dueWithin && task.dueDate) {
      const daysUntilDue = Math.ceil((new Date(task.dueDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24));
      if (daysUntilDue > filters.dueWithin) {
        return false;
      }
    }
    
    // Tags filter
    if (filters.tags && filters.tags.length > 0) {
      const hasMatchingTag = filters.tags.some(tag => task.tags.includes(tag));
      if (!hasMatchingTag) {
        return false;
      }
    }
    
    // Search query filter
    if (filters.searchQuery && filters.searchQuery.trim()) {
      const query = filters.searchQuery.toLowerCase();
      const searchableText = `${task.title} ${task.description || ''} ${task.tags.join(' ')}`.toLowerCase();
      if (!searchableText.includes(query)) {
        return false;
      }
    }
    
    return true;
  });
};

export const useTaskStore = create<TaskStore>()(
  persist(
    (set, get) => ({
      // Initial state
      tasks: [],
      selectedTask: undefined,
      filters: defaultFilters,
      sortOptions: defaultSortOptions,
      viewMode: 'list',
      showCompleted: false,
      groupBy: 'none',
      userStats: defaultStats,
      isLoading: false,
      error: null,
      
      // CRUD Operations
      createTask: async (request: CreateTaskRequest) => {
        set({ isLoading: true, error: null });
        
        try {
          const newTask: Task = {
            id: generateId(),
            ...request,
            tags: request.tags || [], // Ensure tags is always an array
            status: TaskStatus.TODO,
            isSubtask: !!request.parentTaskId,
            subtasks: [],
            createdAt: new Date(),
            updatedAt: new Date(),
          };
          
          const currentTasks = get().tasks;
          
          // If it's a subtask, add it to parent's subtasks array
          if (request.parentTaskId) {
            const updatedTasks = currentTasks.map(task => {
              if (task.id === request.parentTaskId) {
                return {
                  ...task,
                  subtasks: [...task.subtasks, newTask.id],
                  updatedAt: new Date(),
                };
              }
              return task;
            });
            
            set({ 
              tasks: [...updatedTasks, newTask],
              isLoading: false 
            });
          } else {
            set({ 
              tasks: [...currentTasks, newTask],
              isLoading: false 
            });
          }
          
          get().updateUserStats();
          return newTask;
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to create task',
            isLoading: false 
          });
          throw error;
        }
      },
      
      updateTask: async (request: UpdateTaskRequest) => {
        set({ isLoading: true, error: null });
        
        try {
          const currentTasks = get().tasks;
          const taskIndex = currentTasks.findIndex(t => t.id === request.id);
          
          if (taskIndex === -1) {
            throw new Error('Task not found');
          }
          
          const updatedTask: Task = {
            ...currentTasks[taskIndex],
            ...request,
            updatedAt: new Date(),
          };
          
          const updatedTasks = [...currentTasks];
          updatedTasks[taskIndex] = updatedTask;
          
          set({ 
            tasks: updatedTasks,
            isLoading: false,
            selectedTask: get().selectedTask?.id === request.id ? updatedTask : get().selectedTask
          });
          
          get().updateUserStats();
          return updatedTask;
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to update task',
            isLoading: false 
          });
          throw error;
        }
      },
      
      deleteTask: async (taskId: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const currentTasks = get().tasks;
          const taskToDelete = currentTasks.find(t => t.id === taskId);
          
          if (!taskToDelete) {
            throw new Error('Task not found');
          }
          
          // Remove task and all its subtasks
          const tasksToRemove = [taskId, ...taskToDelete.subtasks];
          const updatedTasks = currentTasks.filter(t => !tasksToRemove.includes(t.id));
          
          // Remove from parent's subtasks if it's a subtask
          if (taskToDelete.parentTaskId) {
            const parentIndex = updatedTasks.findIndex(t => t.id === taskToDelete.parentTaskId);
            if (parentIndex !== -1) {
              updatedTasks[parentIndex] = {
                ...updatedTasks[parentIndex],
                subtasks: updatedTasks[parentIndex].subtasks.filter(id => id !== taskId),
                updatedAt: new Date(),
              };
            }
          }
          
          set({ 
            tasks: updatedTasks,
            isLoading: false,
            selectedTask: get().selectedTask?.id === taskId ? undefined : get().selectedTask
          });
          
          get().updateUserStats();
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to delete task',
            isLoading: false 
          });
          throw error;
        }
      },
      
      completeTask: async (taskId: string, focusScore?: number, actualMinutes?: number) => {
        const task = get().tasks.find(t => t.id === taskId);
        if (!task) {
          throw new Error('Task not found');
        }
        
        await get().updateTask({
          id: taskId,
          status: TaskStatus.COMPLETED,
          completedAt: new Date(),
          focusScore,
          actualMinutes,
        });
      },
      
      duplicateTask: async (taskId: string) => {
        const task = get().tasks.find(t => t.id === taskId);
        if (!task) {
          throw new Error('Task not found');
        }
        
        const duplicateRequest: CreateTaskRequest = {
          title: `${task.title} (Copy)`,
          description: task.description,
          priority: task.priority,
          complexity: task.complexity,
          estimatedMinutes: task.estimatedMinutes,
          energyLevel: task.energyLevel,
          category: task.category,
          tags: [...task.tags],
          dueDate: task.dueDate,
        };
        
        return await get().createTask(duplicateRequest);
      },
      
      addSubtask: async (parentId: string, subtaskData: CreateTaskRequest) => {
        return await get().createTask({
          ...subtaskData,
          parentTaskId: parentId,
        });
      },
      
      moveTask: async (taskId: string, newStatus: TaskStatus) => {
        await get().updateTask({
          id: taskId,
          status: newStatus,
        });
      },
      
      // Filtering and sorting
      setFilters: (filters: Partial<TaskFilters>) => {
        set(state => ({
          filters: { ...state.filters, ...filters }
        }));
      },
      
      setSortOptions: (sortOptions: TaskSortOptions) => {
        set({ sortOptions });
      },
      
      setViewMode: (viewMode: 'list' | 'board' | 'calendar') => {
        set({ viewMode });
      },
      
      setGroupBy: (groupBy: 'priority' | 'category' | 'dueDate' | 'none') => {
        set({ groupBy });
      },
      
      clearFilters: () => {
        set({ filters: defaultFilters });
      },
      
      // Computed getters
      getFilteredTasks: () => {
        const { tasks, filters, sortOptions, showCompleted } = get();
        let filteredTasks = filterTasks(tasks, filters);
        
        if (!showCompleted) {
          filteredTasks = filteredTasks.filter(task => task.status !== TaskStatus.COMPLETED);
        }
        
        return sortTasks(filteredTasks, sortOptions);
      },
      
      getTasksByStatus: (status: TaskStatus) => {
        return get().tasks.filter(task => task.status === status);
      },
      
      getOverdueTasks: () => {
        const now = new Date();
        return get().tasks.filter(task => 
          task.dueDate && 
          new Date(task.dueDate) < now && 
          task.status !== TaskStatus.COMPLETED
        );
      },
      
      getTasksForToday: () => {
        const today = new Date();
        const todayStart = new Date(today.getFullYear(), today.getMonth(), today.getDate());
        const todayEnd = new Date(todayStart.getTime() + 24 * 60 * 60 * 1000);
        
        return get().tasks.filter(task => 
          task.dueDate && 
          new Date(task.dueDate) >= todayStart && 
          new Date(task.dueDate) < todayEnd
        );
      },
      
      getSubtasks: (parentId: string) => {
        return get().tasks.filter(task => task.parentTaskId === parentId);
      },
      
      // Stats and analytics
      updateUserStats: () => {
        const { tasks } = get();
        const completedTasks = tasks.filter(task => task.status === TaskStatus.COMPLETED);
        const totalTasks = tasks.length;
        
        const totalPoints = completedTasks.reduce((sum, task) => sum + calculatePoints(task), 0);
        const level = Math.floor(totalPoints / 1000) + 1;
        
        const completionRate = totalTasks > 0 ? (completedTasks.length / totalTasks) * 100 : 0;
        
        const averageFocusScore = completedTasks.length > 0 
          ? completedTasks.reduce((sum, task) => sum + (task.focusScore || 0), 0) / completedTasks.length
          : 0;
        
        // Calculate current streak
        const sortedCompletedTasks = completedTasks
          .sort((a, b) => new Date(b.completedAt!).getTime() - new Date(a.completedAt!).getTime());
        
        let currentStreak = 0;
        let currentDate = new Date();
        currentDate.setHours(0, 0, 0, 0);
        
        for (const task of sortedCompletedTasks) {
          const completedDate = new Date(task.completedAt!);
          completedDate.setHours(0, 0, 0, 0);
          
          if (completedDate.getTime() === currentDate.getTime()) {
            currentStreak++;
            currentDate.setDate(currentDate.getDate() - 1);
          } else {
            break;
          }
        }
        
        const updatedStats: UserTaskStats = {
          totalCompleted: completedTasks.length,
          currentStreak,
          longestStreak: Math.max(get().userStats.longestStreak, currentStreak),
          totalPoints,
          level,
          completionRate,
          averageFocusScore,
        };
        
        set({ userStats: updatedStats });
      },
      
      getCompletionStreak: () => {
        return get().userStats.currentStreak;
      },
      
      // Utility functions
      selectTask: (task: Task | undefined) => {
        set({ selectedTask: task });
      },
      
      searchTasks: (query: string) => {
        return get().tasks.filter(task => {
          const searchText = `${task.title} ${task.description || ''} ${task.tags.join(' ')}`.toLowerCase();
          return searchText.includes(query.toLowerCase());
        });
      },
      
      // Reset and initialization
      resetStore: () => {
        set({
          tasks: [],
          selectedTask: undefined,
          filters: defaultFilters,
          sortOptions: defaultSortOptions,
          viewMode: 'list',
          showCompleted: false,
          groupBy: 'none',
          userStats: defaultStats,
          isLoading: false,
          error: null,
        });
      },
      
      initializeWithSampleData: () => {
        const sampleTasks: Task[] = [
          {
            id: 'sample-1',
            title: 'Complete project proposal',
            description: 'Draft and review the Q4 project proposal document',
            priority: TaskPriority.HIGH,
            complexity: TaskComplexity.COMPLEX,
            estimatedMinutes: 120,
            energyLevel: EnergyLevel.HIGH,
            category: TaskCategory.WORK,
            tags: ['project', 'proposal', 'deadline'],
            status: TaskStatus.TODO,
            isSubtask: false,
            subtasks: [],
            createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
            dueDate: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000),
          },
          {
            id: 'sample-2',
            title: 'Review email inbox',
            description: 'Process and respond to pending emails',
            priority: TaskPriority.MEDIUM,
            complexity: TaskComplexity.SIMPLE,
            estimatedMinutes: 30,
            energyLevel: EnergyLevel.LOW,
            category: TaskCategory.ADMINISTRATIVE,
            tags: ['email', 'communication'],
            status: TaskStatus.TODO,
            isSubtask: false,
            subtasks: [],
            createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
          },
          {
            id: 'sample-3',
            title: 'Exercise - 30 min walk',
            description: 'Take a refreshing walk around the neighborhood',
            priority: TaskPriority.LOW,
            complexity: TaskComplexity.SIMPLE,
            estimatedMinutes: 30,
            energyLevel: EnergyLevel.MEDIUM,
            category: TaskCategory.HEALTH,
            tags: ['exercise', 'wellness', 'outdoor'],
            status: TaskStatus.COMPLETED,
            isSubtask: false,
            subtasks: [],
            createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
            completedAt: new Date(Date.now() - 12 * 60 * 60 * 1000),
            actualMinutes: 35,
            focusScore: 8,
          },
        ];
        
        set({ tasks: sampleTasks });
        get().updateUserStats();
      },
    }),
    {
      name: 'neuroflow-tasks',
      version: 1,
    }
  )
);

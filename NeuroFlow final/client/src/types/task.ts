// Core task management types for neurodivergent-friendly task system

export interface Task {
  id: string;
  title: string;
  description?: string;
  
  // Neurodivergent-friendly properties
  priority: TaskPriority;
  complexity: TaskComplexity;
  estimatedMinutes: number;
  actualMinutes?: number;
  energyLevel: EnergyLevel; // Required energy to complete
  
  // Organization
  category: TaskCategory;
  tags: string[];
  
  // Status and progress
  status: TaskStatus;
  completedAt?: Date;
  createdAt: Date;
  updatedAt: Date;
  dueDate?: Date;
  
  // Task chunking
  isSubtask: boolean;
  parentTaskId?: string;
  subtasks: string[]; // Array of subtask IDs
  
  // Behavioral tracking
  focusScore?: number; // 1-10 how focused user was during task
  interruptionCount?: number;
  completionStreak?: number;
  
  // AI-powered features (for future phases)
  aiGenerated?: boolean;
  difficultyAdjustment?: number; // AI learning adjustment
}

export enum TaskPriority {
  LOW = 'low',
  MEDIUM = 'medium', 
  HIGH = 'high',
  URGENT = 'urgent'
}

export enum TaskComplexity {
  SIMPLE = 'simple',     // Quick, single-step tasks
  MODERATE = 'moderate', // Multi-step but straightforward
  COMPLEX = 'complex',   // Requires planning and focus
  OVERWHELMING = 'overwhelming' // Needs chunking
}

export enum EnergyLevel {
  LOW = 'low',       // Can do when tired/unfocused
  MEDIUM = 'medium', // Requires some focus
  HIGH = 'high'      // Needs peak energy and focus
}

export enum TaskCategory {
  WORK = 'work',
  PERSONAL = 'personal',
  HEALTH = 'health',
  LEARNING = 'learning',
  CREATIVE = 'creative',
  ADMINISTRATIVE = 'administrative',
  SOCIAL = 'social',
  SELF_CARE = 'self_care'
}

export enum TaskStatus {
  TODO = 'todo',
  IN_PROGRESS = 'in_progress', 
  BLOCKED = 'blocked',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled'
}

// Filters and sorting options
export interface TaskFilters {
  status?: TaskStatus[];
  priority?: TaskPriority[];
  category?: TaskCategory[];
  complexity?: TaskComplexity[];
  energyLevel?: EnergyLevel[];
  dueWithin?: number; // days
  tags?: string[];
  searchQuery?: string;
}

export interface TaskSortOptions {
  field: 'priority' | 'dueDate' | 'createdAt' | 'complexity' | 'estimatedMinutes';
  direction: 'asc' | 'desc';
}

// Behavioral analytics (for future phases)
export interface TaskSession {
  id: string;
  taskId: string;
  startTime: Date;
  endTime?: Date;
  focusScore: number;
  interruptionCount: number;
  completed: boolean;
  notes?: string;
}

export interface UserProductivityPattern {
  userId: string;
  peakFocusHours: number[]; // Hours of day when most productive
  preferredTaskComplexity: TaskComplexity[];
  averageTaskDuration: Record<TaskComplexity, number>;
  completionRate: Record<TaskPriority, number>;
  commonBlockers: string[];
  bestCategories: TaskCategory[];
}

// Form interfaces
export interface CreateTaskRequest {
  title: string;
  description?: string;
  priority: TaskPriority;
  complexity: TaskComplexity;
  estimatedMinutes: number;
  energyLevel: EnergyLevel;
  category: TaskCategory;
  tags?: string[];
  dueDate?: Date;
  parentTaskId?: string;
}

export interface UpdateTaskRequest extends Partial<CreateTaskRequest> {
  id: string;
  status?: TaskStatus;
  focusScore?: number;
  interruptionCount?: number;
  actualMinutes?: number;
  completedAt?: Date;
}

// UI state interfaces
export interface TaskViewState {
  selectedTask?: Task;
  filters: TaskFilters;
  sortOptions: TaskSortOptions;
  viewMode: 'list' | 'board' | 'calendar';
  showCompleted: boolean;
  groupBy?: 'priority' | 'category' | 'dueDate' | 'none';
}

// Gamification interfaces (for future phases)
export interface TaskAchievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  unlockedAt?: Date;
  progress: number;
  maxProgress: number;
}

export interface UserTaskStats {
  totalCompleted: number;
  currentStreak: number;
  longestStreak: number;
  totalPoints: number;
  level: number;
  completionRate: number;
  averageFocusScore: number;
}

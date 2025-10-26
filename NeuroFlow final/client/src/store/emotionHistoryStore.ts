import { create } from 'zustand';
import { EmotionType } from '../hooks/useEmotionAnalysis';

export interface EmotionHistoryEntry {
  id: string;
  text: string;
  emotion: EmotionType;
  timestamp: string;
  emotion_spectrum: Record<EmotionType, number>;
}

interface EmotionHistoryStore {
  history: EmotionHistoryEntry[];
  addEntry: (entry: Omit<EmotionHistoryEntry, 'id'>) => void;
  clearHistory: () => void;
}

export const useEmotionHistoryStore = create<EmotionHistoryStore>((set) => ({
  history: [],
  addEntry: (entry) =>
    set((state) => ({
      history: [
        {
          ...entry,
          id: Math.random().toString(36).substring(2),
        },
        ...state.history,
      ],
    })),
  clearHistory: () => set({ history: [] }),
}));
import { useState } from 'react';
import { useMutation, UseMutationResult } from '@tanstack/react-query';

// Log the API URL for debugging
const API_URL = window.location.protocol + '//' + window.location.hostname + ':8000';
console.log('Using API URL:', API_URL);

export type EmotionType = 'anger' | 'disgust' | 'fear' | 'joy' | 'neutral' | 'sadness' | 'surprise' | 'trust';

export interface EmotionResponse {
  emotion: EmotionType;
  probability: number;
  emoji: string;
  emotion_spectrum: Record<EmotionType, number>;
}

export interface EmotionData {
  count: number;
  percentage: number;
  average_intensity: number;
}

export type EmotionDistribution = Record<EmotionType, EmotionData>;

export interface EmotionalInsights {
  total_entries: number;
  emotion_distribution: EmotionDistribution;
  dominant_emotion: EmotionType;
  emotional_volatility: number;
  temporal_patterns?: Record<string, number[]>;
}

export interface EmotionAnalysisHook {
  analyzeEmotion: (text: string) => Promise<EmotionResponse>;
  analyzeBatchEmotions: (texts: string[]) => Promise<EmotionResponse[]>;
  getEmotionalInsights: (startDate: Date, endDate: Date, texts?: string[]) => Promise<EmotionalInsights>;
  isLoading: boolean;
  error: string | null;
  clearError: () => void;
}

interface GetInsightsParams {
  startDate: Date;
  endDate: Date;
  texts?: string[];
}

export function useEmotionAnalysis(): EmotionAnalysisHook {
  const [error, setError] = useState<string | null>(null);

  const {
    mutateAsync: analyzeEmotionMutation,
    isPending: isAnalyzing
  } = useMutation<EmotionResponse, Error, string>({
    mutationFn: async (text: string) => {
      try {
        console.log('Attempting to analyze text:', text);
        console.log('Making request to:', `${API_URL}/api/emotions/analyze`);
        
        const response = await fetch(`${API_URL}/api/emotions/analyze`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ text })
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', [...response.headers.entries()]);
        
        if (!response.ok) {
          const errorText = await response.text();
          console.error('Error response:', errorText);
          try {
            const errorData = JSON.parse(errorText);
            throw new Error(errorData.detail || 'Failed to analyze emotion');
          } catch (e) {
            throw new Error(`Failed to analyze emotion: ${errorText}`);
          }
        }
        
        const data = await response.json();
        console.log('Success response:', data);
        return data;
      } catch (error) {
        console.error('Error analyzing emotion:', error);
        console.error('Error stack:', error.stack);
        throw new Error(`Failed to analyze emotion: ${error.message}`);
      }
    }
  });

  const {
    mutateAsync: analyzeBatchMutation,
    isPending: isAnalyzingBatch
  } = useMutation<EmotionResponse[], Error, string[]>({
    mutationFn: async (texts: string[]) => {
      const response = await fetch(`${API_URL}/api/emotions/analyze-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts }),
      });
      if (!response.ok) throw new Error('Failed to analyze emotions in batch');
      return response.json();
    }
  });

  const {
    mutateAsync: getInsightsMutation,
    isPending: isLoadingInsights
  } = useMutation<EmotionalInsights, Error, GetInsightsParams>({
    mutationFn: async ({ startDate, endDate, texts }: GetInsightsParams) => {
      const response = await fetch(`${API_URL}/api/emotions/insights`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          texts: texts || [],
          time_range: {
            start_date: startDate.toISOString(),
            end_date: endDate.toISOString(),
          },
        }),
      });
      if (!response.ok) throw new Error('Failed to get emotional insights');
      return response.json();
    }
  });

  return {
    analyzeEmotion: analyzeEmotionMutation,
    analyzeBatchEmotions: analyzeBatchMutation,
    getEmotionalInsights: (startDate, endDate, texts) =>
      getInsightsMutation({ startDate, endDate, texts }),
    isLoading: isAnalyzing || isAnalyzingBatch || isLoadingInsights,
    error,
    clearError: () => setError(null),
  };
}

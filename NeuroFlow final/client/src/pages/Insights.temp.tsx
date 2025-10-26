import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Brain, TrendingUp, Calendar } from 'lucide-react';
import { useEmotionAnalysis } from '../hooks/useEmotionAnalysis';
import type { EmotionHistoryEntry } from '../hooks/useEmotionAnalysis';

const Insights = () => {
  const { emotionHistory } = useEmotionAnalysis();

  const calculateEmotionFrequency = () => {
    const frequency: { [key: string]: number } = {};
    emotionHistory.forEach((entry: EmotionHistoryEntry) => {
      frequency[entry.emotion] = (frequency[entry.emotion] || 0) + 1;
    });
    return Object.entries(frequency).map(([emotion, count]) => ({
      emotion,
      count,
      percentage: (count / emotionHistory.length) * 100,
    }));
  };

  const calculateEmotionTrends = () => {
    const trends = emotionHistory.map((entry: EmotionHistoryEntry) => ({
      date: new Date(entry.timestamp).toLocaleDateString(),
      ...entry.emotion_spectrum,
    }));
    return trends;
  };

  const emotionFrequency = calculateEmotionFrequency();
  const emotionTrends = calculateEmotionTrends();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="bg-gradient-to-r from-purple-400 to-purple-500 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <div className="bg-white rounded-full p-3">
              <Brain className="h-8 w-8 text-purple-500" />
            </div>
            <h1 className="text-3xl font-bold text-white">Emotional Insights</h1>
          </div>
          <p className="text-white text-lg opacity-90">
            Explore patterns and trends in your emotional journey. Understanding your emotions
            helps build self-awareness and emotional intelligence.
          </p>
        </div>

        {emotionHistory.length === 0 ? (
          <div className="text-center py-12">
            <h2 className="text-2xl font-semibold text-gray-700 mb-4">
              No emotional data available yet
            </h2>
            <p className="text-gray-600">
              Start by sharing your thoughts in the Emotion Analysis section to see insights here.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center space-x-2 mb-6">
                <TrendingUp className="h-6 w-6 text-purple-500" />
                <h2 className="text-2xl font-semibold text-gray-800">Emotion Distribution</h2>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={emotionFrequency}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="emotion" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="percentage" fill="#9f7aea" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Calendar className="h-6 w-6 text-purple-500" />
                <h2 className="text-2xl font-semibold text-gray-800">Emotion Trends</h2>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={emotionTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    {Object.keys(emotionTrends[0] || {})
                      .filter((key) => key !== 'date')
                      .map((emotion, index) => (
                        <Bar
                          key={emotion}
                          dataKey={emotion}
                          fill={`hsl(${index * 40}, 70%, 60%)`}
                          stackId="emotions"
                        />
                      ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 lg:col-span-2">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6">Emotional Summary</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-purple-50 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-700 mb-2">Most Common Emotion</h3>
                  <p className="text-3xl font-bold text-purple-600">
                    {emotionFrequency[0]?.emotion || 'N/A'}
                  </p>
                  <p className="text-sm text-gray-600">
                    {emotionFrequency[0]?.percentage.toFixed(1)}% of entries
                  </p>
                </div>
                <div className="bg-purple-50 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-700 mb-2">Total Entries</h3>
                  <p className="text-3xl font-bold text-purple-600">{emotionHistory.length}</p>
                  <p className="text-sm text-gray-600">emotional check-ins</p>
                </div>
                <div className="bg-purple-50 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-700 mb-2">Last Check-in</h3>
                  <p className="text-3xl font-bold text-purple-600">
                    {emotionHistory[0]
                      ? new Date(emotionHistory[0].timestamp).toLocaleDateString()
                      : 'N/A'}
                  </p>
                  <p className="text-sm text-gray-600">most recent entry</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Insights;

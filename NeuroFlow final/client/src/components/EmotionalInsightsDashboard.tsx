import React, { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import { useEmotionAnalysis } from '../hooks/useEmotionAnalysis';

import { EmotionType, EmotionData, EmotionDistribution } from '../hooks/useEmotionAnalysis';

const COLORS: { [key in EmotionType]: string } = {
  anger: '#ff4d4d',
  disgust: '#8b4513',
  fear: '#800080',
  joy: '#ffd700',
  neutral: '#808080',
  sadness: '#4169e1',
  surprise: '#ff69b4',
  trust: '#98fb98',
};

const EmotionalInsightsDashboard: React.FC = () => {
  const { getEmotionalInsights, isLoading, error } = useEmotionAnalysis();
  const [insights, setInsights] = useState<any>(null);
  const [timeRange, setTimeRange] = useState('week');

  useEffect(() => {
    const fetchInsights = async () => {
      const endDate = new Date();
      const startDate = new Date();

      switch (timeRange) {
        case 'week':
          startDate.setDate(startDate.getDate() - 7);
          break;
        case 'month':
          startDate.setDate(startDate.getDate() - 30);
          break;
        case 'year':
          startDate.setDate(startDate.getDate() - 365);
          break;
      }

      const data = await getEmotionalInsights(startDate, endDate);
      setInsights(data);
    };

    fetchInsights();
  }, [timeRange]);

  const emotionDistributionData = useMemo(() => {
    if (!insights?.emotion_distribution) return [];
    return Object.entries(insights.emotion_distribution).map(([emotion, data]) => ({
      name: emotion as EmotionType,
      value: (data as EmotionData).percentage,
      count: (data as EmotionData).count,
      intensity: (data as EmotionData).average_intensity,
    }));
  }, [insights]);

  const temporalData = useMemo(() => {
    if (!insights?.temporal_patterns) return [];
    const dates = Object.keys(insights.temporal_patterns);
    return dates.map((date) => ({
      date,
      ...Object.entries(insights.temporal_patterns[date]).reduce(
        (acc, [emotion, value]) => ({ ...acc, [emotion]: value }),
        {}
      ),
    }));
  }, [insights]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-700">Error loading insights: {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 p-6">
      {/* Time Range Selector */}
      <div className="flex justify-end space-x-4">
        {['week', 'month', 'year'].map((range) => (
          <button
            key={range}
            onClick={() => setTimeRange(range)}
            className={`px-4 py-2 rounded-lg ${
              timeRange === range
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {range.charAt(0).toUpperCase() + range.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Emotion Distribution */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Emotion Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={emotionDistributionData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ name, value }) => `${name} (${value.toFixed(1)}%)`}
              >
                {emotionDistributionData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[entry.name as EmotionType]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Emotional Intensity */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Emotional Intensity</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={emotionDistributionData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="name" />
              <PolarRadiusAxis />
              <Radar dataKey="intensity" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Temporal Patterns */}
        <div className="bg-white rounded-xl shadow-lg p-6 col-span-1 md:col-span-2">
          <h3 className="text-lg font-semibold mb-4">Emotional Trends Over Time</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={temporalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(COLORS).map((emotion) => (
                <Line
                  key={emotion}
                  type="monotone"
                  dataKey={emotion}
                  stroke={COLORS[emotion as EmotionType]}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Summary Stats */}
        <div className="bg-white rounded-xl shadow-lg p-6 col-span-1 md:col-span-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <h4 className="text-gray-600">Dominant Emotion</h4>
              <p className="text-2xl font-bold mt-2">
                {insights?.dominant_emotion}{' '}
                {insights?.dominant_emotion && COLORS[insights.dominant_emotion as EmotionType] && (
                  <span role="img" aria-label={insights.dominant_emotion}>
                    {insights.emoji_map?.[insights.dominant_emotion]}
                  </span>
                )}
              </p>
            </div>
            <div className="text-center">
              <h4 className="text-gray-600">Emotional Volatility</h4>
              <p className="text-2xl font-bold mt-2">
                {(insights?.emotional_volatility * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center">
              <h4 className="text-gray-600">Total Entries Analyzed</h4>
              <p className="text-2xl font-bold mt-2">{insights?.total_entries}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmotionalInsightsDashboard;
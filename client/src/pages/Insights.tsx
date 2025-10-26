import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Heart, 
  Clock, 
  Calendar, 
  Target,
  Brain,
  Activity,
  Zap,
  Award,
  Sun,
  Moon,
  Coffee,
  Smile
} from 'lucide-react';

interface EmotionData {
  date: string;
  primary_emotion: string;
  sentiment: string;
  intensity: number;
  emotions: { [key: string]: number };
}

interface ActivityData {
  date: string;
  tasks_completed: number;
  total_tasks: number;
  productivity_score: number;
  focus_time: number;
  break_time: number;
}

interface InsightMetric {
  label: string;
  value: string;
  change: number;
  icon: React.ReactNode;
  color: string;
}

const Insights: React.FC = () => {
  const [emotionData, setEmotionData] = useState<EmotionData[]>([]);
  const [activityData, setActivityData] = useState<ActivityData[]>([]);
  const [insights, setInsights] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('7d');

  useEffect(() => {
    fetchInsightsData();
  }, [timeRange]);

  const fetchInsightsData = async () => {
    setLoading(true);
    try {
      const [emotionResponse, activityResponse, insightsResponse] = await Promise.all([
        fetch(`/api/insights/emotions?range=${timeRange}`),
        fetch(`/api/insights/activity?range=${timeRange}`),
        fetch(`/api/insights/summary?range=${timeRange}`)
      ]);

      if (emotionResponse.ok) {
        const emotionResult = await emotionResponse.json();
        setEmotionData(emotionResult.data || []);
      }

      if (activityResponse.ok) {
        const activityResult = await activityResponse.json();
        setActivityData(activityResult.data || []);
      }

      if (insightsResponse.ok) {
        const insightsResult = await insightsResponse.json();
        setInsights(insightsResult);
      }
    } catch (error) {
      console.error('Error fetching insights:', error);
      // Generate mock data for demo
      generateMockData();
    } finally {
      setLoading(false);
    }
  };

  const generateMockData = () => {
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
    const emotions = ['joy', 'calm', 'excited', 'focused', 'peaceful', 'motivated'];
    const sentiments = ['positive', 'neutral', 'negative'];
    
    const mockEmotionData: EmotionData[] = [];
    const mockActivityData: ActivityData[] = [];

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      mockEmotionData.push({
        date: date.toISOString().split('T')[0],
        primary_emotion: emotions[Math.floor(Math.random() * emotions.length)],
        sentiment: sentiments[Math.floor(Math.random() * sentiments.length)],
        intensity: 0.3 + Math.random() * 0.7,
        emotions: {
          joy: Math.random(),
          calm: Math.random(),
          excited: Math.random(),
          focused: Math.random(),
          peaceful: Math.random(),
          motivated: Math.random(),
        }
      });

      mockActivityData.push({
        date: date.toISOString().split('T')[0],
        tasks_completed: Math.floor(Math.random() * 8) + 2,
        total_tasks: Math.floor(Math.random() * 5) + 8,
        productivity_score: 0.5 + Math.random() * 0.5,
        focus_time: Math.floor(Math.random() * 240) + 60,
        break_time: Math.floor(Math.random() * 60) + 15,
      });
    }

    setEmotionData(mockEmotionData);
    setActivityData(mockActivityData);
    
    // Generate mock insights
    setInsights({
      emotional_wellness: 0.78,
      productivity_trend: 0.15,
      focus_improvement: 0.23,
      stress_level: 0.32,
      sleep_quality: 0.85,
      energy_levels: 0.71,
    });
  };

  const getKeyMetrics = (): InsightMetric[] => {
    if (!insights) return [];

    return [
      {
        label: 'Emotional Wellness',
        value: `${(insights.emotional_wellness * 100).toFixed(0)}%`,
        change: insights.emotional_wellness > 0.7 ? 12 : -5,
        icon: <Heart className="w-6 h-6" />,
        color: 'text-pink-500'
      },
      {
        label: 'Productivity Score',
        value: `${(insights.productivity_trend * 100 + 60).toFixed(0)}%`,
        change: insights.productivity_trend * 100,
        icon: <Target className="w-6 h-6" />,
        color: 'text-blue-500'
      },
      {
        label: 'Focus Time',
        value: `${Math.floor(insights.focus_improvement * 100 + 180)}min`,
        change: insights.focus_improvement * 100,
        icon: <Brain className="w-6 h-6" />,
        color: 'text-purple-500'
      },
      {
        label: 'Energy Level',
        value: `${(insights.energy_levels * 100).toFixed(0)}%`,
        change: insights.energy_levels > 0.6 ? 8 : -3,
        icon: <Zap className="w-6 h-6" />,
        color: 'text-yellow-500'
      }
    ];
  };

  const getEmotionTrend = () => {
    if (emotionData.length === 0) return [];
    
    const emotionCounts: { [key: string]: number } = {};
    emotionData.forEach(data => {
      emotionCounts[data.primary_emotion] = (emotionCounts[data.primary_emotion] || 0) + 1;
    });

    return Object.entries(emotionCounts)
      .map(([emotion, count]) => ({
        emotion,
        count,
        percentage: (count / emotionData.length) * 100
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  };

  const getProductivityTrend = () => {
    return activityData.map(data => ({
      date: new Date(data.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      productivity: data.productivity_score * 100,
      tasksCompleted: data.tasks_completed,
      focusTime: data.focus_time
    }));
  };

  if (loading) {
    return (
      <div className="space-y-8 animate-gentle-fade-in">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="text-text-secondary">Analyzing your patterns...</p>
          </div>
        </div>
      </div>
    );
  }

  const keyMetrics = getKeyMetrics();
  const emotionTrend = getEmotionTrend();
  const productivityTrend = getProductivityTrend();

  return (
    <div className="space-y-8 animate-gentle-fade-in pb-20 md:pb-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
        <div className="space-y-2">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-full p-3">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-text-primary">Insights Dashboard</h1>
          </div>
          <p className="text-text-secondary">
            Deep insights into your emotional patterns and productivity trends
          </p>
        </div>

        {/* Time Range Selector */}
        <div className="flex bg-surface rounded-lg p-1 border border-border">
          {(['7d', '30d', '90d'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                timeRange === range
                  ? 'bg-primary text-white shadow-sm'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background'
              }`}
            >
              {range === '7d' ? 'Last 7 days' : range === '30d' ? 'Last 30 days' : 'Last 90 days'}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {keyMetrics.map((metric, index) => (
          <div key={index} className="bg-surface rounded-2xl p-6 shadow-lg border border-border">
            <div className="flex items-center justify-between">
              <div className="space-y-2">
                <p className="text-text-secondary text-sm font-medium">{metric.label}</p>
                <p className="text-2xl font-bold text-text-primary">{metric.value}</p>
                <div className={`flex items-center space-x-1 text-sm ${
                  metric.change >= 0 ? 'text-green-500' : 'text-red-500'
                }`}>
                  <TrendingUp className={`w-4 h-4 ${metric.change < 0 ? 'rotate-180' : ''}`} />
                  <span>{metric.change >= 0 ? '+' : ''}{metric.change.toFixed(1)}%</span>
                </div>
              </div>
              <div className={`${metric.color} bg-current bg-opacity-10 rounded-full p-3`}>
                {metric.icon}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Emotion Analysis Chart */}
        <div className="bg-surface rounded-2xl p-6 shadow-lg border border-border">
          <div className="space-y-6">
            <div className="flex items-center space-x-3">
              <Heart className="w-6 h-6 text-pink-500" />
              <h3 className="text-xl font-semibold text-text-primary">Emotion Patterns</h3>
            </div>

            <div className="space-y-4">
              {emotionTrend.map((item, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-text-primary font-medium capitalize">
                      {item.emotion}
                    </span>
                    <span className="text-text-secondary text-sm">
                      {item.percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-background rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-pink-500 to-purple-500 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${item.percentage}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Productivity Chart */}
        <div className="bg-surface rounded-2xl p-6 shadow-lg border border-border">
          <div className="space-y-6">
            <div className="flex items-center space-x-3">
              <Activity className="w-6 h-6 text-blue-500" />
              <h3 className="text-xl font-semibold text-text-primary">Productivity Trends</h3>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="space-y-1">
                  <p className="text-2xl font-bold text-green-500">
                    {activityData.reduce((sum, d) => sum + d.tasks_completed, 0)}
                  </p>
                  <p className="text-text-secondary text-sm">Tasks Done</p>
                </div>
                <div className="space-y-1">
                  <p className="text-2xl font-bold text-blue-500">
                    {Math.floor(activityData.reduce((sum, d) => sum + d.focus_time, 0) / activityData.length)}m
                  </p>
                  <p className="text-text-secondary text-sm">Avg Focus</p>
                </div>
                <div className="space-y-1">
                  <p className="text-2xl font-bold text-purple-500">
                    {Math.floor(activityData.reduce((sum, d) => sum + d.productivity_score, 0) / activityData.length * 100)}%
                  </p>
                  <p className="text-text-secondary text-sm">Efficiency</p>
                </div>
              </div>

              <div className="space-y-3">
                {productivityTrend.slice(-5).map((item, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-text-secondary text-sm">{item.date}</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-background rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                          style={{ width: `${item.productivity}%` }}
                        ></div>
                      </div>
                      <span className="text-text-primary text-sm font-medium">
                        {item.productivity.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AI Insights & Recommendations */}
      <div className="bg-surface rounded-2xl p-6 shadow-lg border border-border">
        <div className="space-y-6">
          <div className="flex items-center space-x-3">
            <Brain className="w-6 h-6 text-purple-500" />
            <h3 className="text-xl font-semibold text-text-primary">AI-Powered Insights</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-semibold text-text-primary flex items-center space-x-2">
                <Smile className="w-5 h-5 text-yellow-500" />
                <span>Emotional Patterns</span>
              </h4>
              <div className="space-y-3 text-text-secondary">
                <p>• You show strongest positive emotions in the morning hours</p>
                <p>• Focus and calm emotions peak during mid-week</p>
                <p>• Stress levels tend to increase before deadlines</p>
                <p>• Taking breaks improves your emotional balance by 23%</p>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-semibold text-text-primary flex items-center space-x-2">
                <Target className="w-5 h-5 text-blue-500" />
                <span>Productivity Insights</span>
              </h4>
              <div className="space-y-3 text-text-secondary">
                <p>• Your peak productivity window is 9-11 AM</p>
                <p>• Task completion rate improves 15% with morning planning</p>
                <p>• Short breaks every 45 minutes boost focus by 30%</p>
                <p>• You work most efficiently on Tuesday and Wednesday</p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-primary/10 to-purple-600/10 rounded-xl p-4 border border-primary/20">
            <h4 className="font-semibold text-text-primary mb-2 flex items-center space-x-2">
              <Award className="w-5 h-5 text-primary" />
              <span>Personalized Recommendations</span>
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="text-text-primary font-medium">🌅 Morning Routine</p>
                <p className="text-text-secondary text-sm">
                  Start with 5 minutes of mindfulness to boost positive emotions
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-text-primary font-medium">🎯 Focus Sessions</p>
                <p className="text-text-secondary text-sm">
                  Schedule demanding tasks during your 9-11 AM peak window
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-text-primary font-medium">⏰ Break Timer</p>
                <p className="text-text-secondary text-sm">
                  Set 45-minute focused work blocks with 10-minute breaks
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-text-primary font-medium">📊 Weekly Planning</p>
                <p className="text-text-secondary text-sm">
                  Plan important tasks for Tuesday-Wednesday when you're most efficient
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Insights;
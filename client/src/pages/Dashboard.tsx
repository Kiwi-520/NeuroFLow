import React from 'react';
import { 
  Brain, 
  Target, 
  TrendingUp, 
  Clock, 
  CheckCircle, 
  Calendar,
  Smile,
  Star
} from 'lucide-react';

const Dashboard: React.FC = () => {
  // Mock data for demonstration
  const stats = [
    { label: 'Tasks Completed', value: '12', icon: CheckCircle, color: 'text-green-500' },
    { label: 'Focus Time', value: '2h 30m', icon: Clock, color: 'text-blue-500' },
    { label: 'Streak Days', value: '7', icon: TrendingUp, color: 'text-purple-500' },
    { label: 'Mood Score', value: '8.2', icon: Smile, color: 'text-yellow-500' },
  ];

  const recentActivities = [
    { action: 'Completed morning routine', time: '2 hours ago', type: 'success' },
    { action: 'Started focus session', time: '3 hours ago', type: 'info' },
    { action: 'Updated mood check-in', time: '5 hours ago', type: 'neutral' },
    { action: 'Set daily goals', time: '1 day ago', type: 'success' },
  ];

  return (
    <div className="space-y-8">
      {/* Welcome Header */}
      <div className="bg-primary rounded-2xl p-6 text-white shadow-md">
        <div className="flex items-center space-x-4">
          <div className="bg-white/20 rounded-full p-3">
            <Brain className="h-8 w-8" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Welcome back!</h1>
            <p className="text-white/80">
              You're doing great. Let's make today amazing! ✨
            </p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.label}
              className="bg-surface rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow duration-200 border border-border"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-text-secondary text-sm font-medium">
                    {stat.label}
                  </p>
                  <p className="text-2xl font-bold text-text-primary mt-1">
                    {stat.value}
                  </p>
                </div>
                <div className="text-primary">
                  <Icon size={24} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Today's Focus */}
        <div className="lg:col-span-2 bg-surface rounded-xl p-6 shadow-sm border border-border">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-text-primary flex items-center space-x-2">
              <Target className="h-5 w-5 text-primary" />
              <span>Today's Focus</span>
            </h2>
            <button className="text-primary hover:text-secondary text-sm font-medium">
              View All
            </button>
          </div>

          <div className="space-y-4">
            <div className="flex items-center space-x-4 p-4 bg-primary/10 rounded-lg border border-primary/20">
              <div className="flex-shrink-0">
                <div className="w-3 h-3 bg-primary rounded-full"></div>
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-text-primary">
                  Complete project proposal
                </h3>
                <p className="text-sm text-text-secondary">
                  Break into 3 smaller tasks • Due in 2 hours
                </p>
              </div>
              <Star className="h-5 w-5 text-accent" />
            </div>

            <div className="flex items-center space-x-4 p-4 bg-surface border border-border rounded-lg">
              <div className="flex-shrink-0">
                <div className="w-3 h-3 bg-text-secondary rounded-full"></div>
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-text-primary">
                  Take mindfulness break
                </h3>
                <p className="text-sm text-text-secondary">
                  5-minute breathing exercise
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4 p-4 bg-surface border border-border rounded-lg">
              <div className="flex-shrink-0">
                <div className="w-3 h-3 bg-text-secondary rounded-full"></div>
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-text-primary">
                  Review daily goals
                </h3>
                <p className="text-sm text-text-secondary">
                  Evening reflection time
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activities */}
        <div className="bg-surface rounded-xl p-6 shadow-sm border border-border">
          <h2 className="text-xl font-semibold text-text-primary mb-6 flex items-center space-x-2">
            <Calendar className="h-5 w-5 text-primary" />
            <span>Recent Activities</span>
          </h2>

          <div className="space-y-4">
            {recentActivities.map((activity, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                  activity.type === 'success' ? 'bg-emotion-success' :
                  activity.type === 'info' ? 'bg-primary' : 'bg-text-secondary'
                }`}></div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-text-primary">
                    {activity.action}
                  </p>
                  <p className="text-xs text-text-secondary mt-1">
                    {activity.time}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-surface rounded-xl p-6 shadow-sm border border-border">
        <h2 className="text-xl font-semibold text-text-primary mb-6">
          Quick Actions
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button className="p-4 bg-primary/10 hover:bg-primary/20 border border-primary/20 rounded-lg transition-colors duration-200 text-center">
            <div className="text-2xl mb-2">🎯</div>
            <div className="text-sm font-medium text-text-primary">
              Add Task
            </div>
          </button>
          <button className="p-4 bg-secondary/10 hover:bg-secondary/20 border border-secondary/20 rounded-lg transition-colors duration-200 text-center">
            <div className="text-2xl mb-2">😊</div>
            <div className="text-sm font-medium text-text-primary">
              Mood Check
            </div>
          </button>
          <button className="p-4 bg-accent/10 hover:bg-accent/20 border border-accent/20 rounded-lg transition-colors duration-200 text-center">
            <div className="text-2xl mb-2">⏰</div>
            <div className="text-sm font-medium text-text-primary">
              Focus Timer
            </div>
          </button>
          <button className="p-4 bg-primary/10 hover:bg-primary/20 border border-primary/20 rounded-lg transition-colors duration-200 text-center">
            <div className="text-2xl mb-2">🧩</div>
            <div className="text-sm font-medium text-text-primary">
              Brain Games
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
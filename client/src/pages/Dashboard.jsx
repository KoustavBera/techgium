import React from 'react';
import StatsCard from '../components/StatsCard';
import ActivityFeed from '../components/ActivityFeed';
import QuickActions from '../components/QuickActions';

const Dashboard = () => {
  const stats = [
    { title: 'Total Assessments', value: '12', color: 'blue', icon: '📊' },
    { title: 'Health Score', value: '85/100', color: 'green', icon: '❤️' },
    { title: 'Last Check', value: '2 days ago', color: 'yellow', icon: '⏰' },
    { title: 'Pending Reports', value: '3', color: 'red', icon: '📋' }
  ];

  const activities = [
    { text: 'Completed comprehensive health assessment', time: '2 hours ago', type: 'success' },
    { text: 'Uploaded lab report for blood work', time: '1 day ago', type: 'info' },
    { text: 'Started new chat session with AI assistant', time: '2 days ago', type: 'primary' },
    { text: 'Downloaded health report PDF', time: '3 days ago', type: 'secondary' }
  ];

  const quickActions = [
    { name: 'Start Health Chat', icon: '💬', description: 'Begin AI-powered health assessment' },
    { name: 'Upload Lab Report', icon: '📄', description: 'Add new medical documents' },
    { name: 'View Reports', icon: '📊', description: 'Access your health analytics' },
    { name: 'Schedule Checkup', icon: '📅', description: 'Book your next appointment' }
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Health Dashboard</h1>
          <p className="mt-1 text-sm text-gray-600">
            Welcome back! Here's your health overview for today.
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <button className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium shadow-sm">
            🩺 Start New Assessment
          </button>
        </div>
      </div>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <StatsCard key={index} {...stat} />
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Activity Feed - Takes 2 columns */}
        <div className="lg:col-span-2">
          <ActivityFeed activities={activities} />
        </div>
        
        {/* Quick Actions - Takes 1 column */}
        <div>
          <QuickActions actions={quickActions} />
        </div>
      </div>

      {/* Health Insights */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-100">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <span className="text-2xl">🧠</span>
            </div>
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Health Insights</h3>
            <p className="text-gray-600 mb-4">
              Based on your recent assessments, your overall health trend is positive. 
              Consider scheduling a follow-up for your cardiovascular health monitoring.
            </p>
            <button className="text-blue-600 hover:text-blue-800 font-medium">
              View Detailed Analysis →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

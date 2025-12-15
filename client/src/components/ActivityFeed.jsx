import React from 'react';

const ActivityFeed = ({ activities }) => {
  const getActivityColor = (type) => {
    const colors = {
      success: 'bg-green-100 text-green-800',
      info: 'bg-blue-100 text-blue-800',
      primary: 'bg-indigo-100 text-indigo-800',
      secondary: 'bg-gray-100 text-gray-800'
    };
    return colors[type] || colors.secondary;
  };

  const getActivityIcon = (type) => {
    const icons = {
      success: '✅',
      info: 'ℹ️',
      primary: '🔵',
      secondary: '📄'
    };
    return icons[type] || icons.secondary;
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100">
      <div className="p-6 border-b border-gray-100">
        <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
        <p className="text-sm text-gray-600 mt-1">Your latest health-related actions</p>
      </div>
      <div className="p-6">
        <div className="space-y-4">
          {activities.map((activity, index) => (
            <div key={index} className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getActivityColor(activity.type)}`}>
                  <span className="text-sm">{getActivityIcon(activity.type)}</span>
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">{activity.text}</p>
                <p className="text-xs text-gray-500 mt-1">{activity.time}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-6 pt-4 border-t border-gray-100">
          <button className="text-sm text-blue-600 hover:text-blue-800 font-medium">
            View all activity →
          </button>
        </div>
      </div>
    </div>
  );
};

export default ActivityFeed;
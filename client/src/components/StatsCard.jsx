import React from 'react';

const StatsCard = ({ title, value, color, icon }) => {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-50 border-blue-200',
    green: 'text-green-600 bg-green-50 border-green-200',
    yellow: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    red: 'text-red-600 bg-red-50 border-red-200'
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className={`text-2xl font-bold ${colorClasses[color]?.split(' ')[0] || 'text-gray-900'}`}>
            {value}
          </p>
        </div>
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${colorClasses[color]?.split(' ').slice(1).join(' ') || 'bg-gray-50'}`}>
          <span className="text-xl">{icon}</span>
        </div>
      </div>
    </div>
  );
};

export default StatsCard;
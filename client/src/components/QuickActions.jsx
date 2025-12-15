import React from 'react';

const QuickActions = ({ actions }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100">
      <div className="p-6 border-b border-gray-100">
        <h3 className="text-lg font-semibold text-gray-900">Quick Actions</h3>
        <p className="text-sm text-gray-600 mt-1">Common tasks and shortcuts</p>
      </div>
      <div className="p-6">
        <div className="space-y-3">
          {actions.map((action, index) => (
            <button 
              key={index} 
              className="w-full flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-all group"
            >
              <div className="flex items-center space-x-3">
                <span className="text-xl">{action.icon}</span>
                <div className="text-left">
                  <p className="text-sm font-medium text-gray-900 group-hover:text-blue-600">
                    {action.name}
                  </p>
                  <p className="text-xs text-gray-500">{action.description}</p>
                </div>
              </div>
              <svg className="w-4 h-4 text-gray-400 group-hover:text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QuickActions;
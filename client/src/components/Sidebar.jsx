import React from 'react';

const Sidebar = ({ activeTab, setActiveTab, sidebarOpen, setSidebarOpen }) => {
  const navItems = [
    { id: 'dashboard', name: 'Dashboard', icon: '📊' },
    { id: 'chat', name: 'Health Chat', icon: '💬' },
    { id: 'history', name: 'History', icon: '📋' },
    { id: 'reports', name: 'Reports', icon: '📄' },
    { id: 'profile', name: 'Profile', icon: '👤' },
    { id: 'settings', name: 'Settings', icon: '⚙️' }
  ];

  return (
    <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0`}>
      {/* Brand */}
      <div className="flex items-center justify-center h-16 bg-blue-600">
        <h1 className="text-xl font-bold text-white">Chiranjeevi</h1>
      </div>
      
      {/* Navigation */}
      <nav className="mt-8">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => {
              setActiveTab(item.id);
              setSidebarOpen(false);
            }}
            className={`w-full flex items-center px-6 py-3 text-left hover:bg-blue-50 transition-colors ${
              activeTab === item.id ? 'bg-blue-50 text-blue-600 border-r-2 border-blue-600' : 'text-gray-700'
            }`}
          >
            <span className="mr-3 text-lg">{item.icon}</span>
            {item.name}
          </button>
        ))}
      </nav>
      
      {/* User section */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
        <div className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50">
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
            <span className="text-white text-sm font-medium">U</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">User</p>
            <p className="text-xs text-gray-500 truncate">Health Assistant</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
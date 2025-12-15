import React, { useState } from 'react';

const Settings = () => {
  const [settings, setSettings] = useState({
    notifications: {
      email: true,
      push: false,
      sms: false,
      assessmentReminders: true,
      reportReady: true,
      emergencyAlerts: true
    },
    privacy: {
      profileVisibility: 'private',
      dataRetention: '2years',
      analyticsOptOut: false,
      marketingOptOut: true
    },
    appearance: {
      theme: 'light',
      language: 'en',
      dateFormat: 'MM/DD/YYYY',
      timeFormat: '12hour'
    }
  });

  const [activeSection, setActiveSection] = useState('notifications');

  const handleSettingChange = (section, key, value) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  const handleSave = () => {
    // TODO: Save settings to backend
    console.log('Saving settings:', settings);
  };

  const sections = [
    { id: 'notifications', name: 'Notifications', icon: '🔔', description: 'Manage your notification preferences' },
    { id: 'privacy', name: 'Privacy & Security', icon: '🔒', description: 'Control your data and privacy settings' },
    { id: 'appearance', name: 'Appearance', icon: '🎨', description: 'Customize the look and feel' },
    { id: 'account', name: 'Account', icon: '👤', description: 'Account management and data' }
  ];

  const ToggleSwitch = ({ enabled, onChange, disabled = false }) => (
    <button
      onClick={() => !disabled && onChange(!enabled)}
      disabled={disabled}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
        enabled ? 'bg-blue-600' : 'bg-gray-200'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          enabled ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="mt-1 text-sm text-gray-600">
            Manage your application preferences and account settings
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <button
            onClick={() => setSettings({})}
            className="bg-white text-gray-700 border border-gray-300 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors"
          >
            🔄 Reset to Defaults
          </button>
          <button
            onClick={handleSave}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            💾 Save Changes
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
            <nav className="space-y-2">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-start p-3 text-left rounded-lg transition-colors ${
                    activeSection === section.id
                      ? 'bg-blue-50 text-blue-700 border border-blue-200'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`}
                >
                  <span className="mr-3 text-lg">{section.icon}</span>
                  <div>
                    <p className="font-medium">{section.name}</p>
                    <p className="text-xs text-gray-500 mt-1">{section.description}</p>
                  </div>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            
            {/* Notifications Settings */}
            {activeSection === 'notifications' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Notification Preferences</h3>
                  <p className="text-sm text-gray-600 mb-6">
                    Choose how you want to be notified about important updates and events.
                  </p>
                </div>

                <div className="space-y-6">
                  <div className="border-b border-gray-100 pb-6">
                    <h4 className="text-md font-medium text-gray-900 mb-4">Delivery Methods</h4>
                    <div className="space-y-4">
                      {[
                        { key: 'email', label: 'Email Notifications', desc: 'Receive notifications via email' },
                        { key: 'push', label: 'Push Notifications', desc: 'Browser push notifications' },
                        { key: 'sms', label: 'SMS Notifications', desc: 'Text messages for urgent updates' }
                      ].map(({ key, label, desc }) => (
                        <div key={key} className="flex items-center justify-between">
                          <div>
                            <h5 className="text-sm font-medium text-gray-900">{label}</h5>
                            <p className="text-sm text-gray-500">{desc}</p>
                          </div>
                          <ToggleSwitch
                            enabled={settings.notifications[key]}
                            onChange={(value) => handleSettingChange('notifications', key, value)}
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-md font-medium text-gray-900 mb-4">Notification Types</h4>
                    <div className="space-y-4">
                      {[
                        { key: 'assessmentReminders', label: 'Assessment Reminders', desc: 'Regular health check reminders' },
                        { key: 'reportReady', label: 'Report Ready', desc: 'When health reports are available' },
                        { key: 'emergencyAlerts', label: 'Emergency Alerts', desc: 'Critical health notifications' }
                      ].map(({ key, label, desc }) => (
                        <div key={key} className="flex items-center justify-between">
                          <div>
                            <h5 className="text-sm font-medium text-gray-900">{label}</h5>
                            <p className="text-sm text-gray-500">{desc}</p>
                          </div>
                          <ToggleSwitch
                            enabled={settings.notifications[key]}
                            onChange={(value) => handleSettingChange('notifications', key, value)}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Privacy Settings */}
            {activeSection === 'privacy' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Privacy & Security</h3>
                  <p className="text-sm text-gray-600 mb-6">
                    Control how your data is used and shared.
                  </p>
                </div>

                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Profile Visibility</label>
                    <select
                      value={settings.privacy.profileVisibility}
                      onChange={(e) => handleSettingChange('privacy', 'profileVisibility', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="private">Private</option>
                      <option value="public">Public</option>
                      <option value="friends">Friends Only</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Data Retention Period</label>
                    <select
                      value={settings.privacy.dataRetention}
                      onChange={(e) => handleSettingChange('privacy', 'dataRetention', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="1year">1 Year</option>
                      <option value="2years">2 Years</option>
                      <option value="5years">5 Years</option>
                      <option value="forever">Forever</option>
                    </select>
                  </div>

                  <div className="space-y-4">
                    {[
                      { key: 'analyticsOptOut', label: 'Analytics Opt-out', desc: 'Prevent usage analytics collection' },
                      { key: 'marketingOptOut', label: 'Marketing Opt-out', desc: 'No marketing communications' }
                    ].map(({ key, label, desc }) => (
                      <div key={key} className="flex items-center justify-between">
                        <div>
                          <h4 className="text-sm font-medium text-gray-900">{label}</h4>
                          <p className="text-sm text-gray-500">{desc}</p>
                        </div>
                        <ToggleSwitch
                          enabled={settings.privacy[key]}
                          onChange={(value) => handleSettingChange('privacy', key, value)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Appearance Settings */}
            {activeSection === 'appearance' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Appearance</h3>
                  <p className="text-sm text-gray-600 mb-6">
                    Customize the look and feel of your application.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Theme</label>
                    <select
                      value={settings.appearance.theme}
                      onChange={(e) => handleSettingChange('appearance', 'theme', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                      <option value="auto">Auto (System)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Language</label>
                    <select
                      value={settings.appearance.language}
                      onChange={(e) => handleSettingChange('appearance', 'language', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="en">English</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                      <option value="de">German</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Date Format</label>
                    <select
                      value={settings.appearance.dateFormat}
                      onChange={(e) => handleSettingChange('appearance', 'dateFormat', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="MM/DD/YYYY">MM/DD/YYYY</option>
                      <option value="DD/MM/YYYY">DD/MM/YYYY</option>
                      <option value="YYYY-MM-DD">YYYY-MM-DD</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Time Format</label>
                    <select
                      value={settings.appearance.timeFormat}
                      onChange={(e) => handleSettingChange('appearance', 'timeFormat', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="12hour">12 Hour</option>
                      <option value="24hour">24 Hour</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* Account Settings */}
            {activeSection === 'account' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Management</h3>
                  <p className="text-sm text-gray-600 mb-6">
                    Manage your account data and preferences.
                  </p>
                </div>

                <div className="space-y-4">
                  {[
                    { title: 'Change Password', desc: 'Update your account password', icon: '🔑', color: 'blue' },
                    { title: 'Two-Factor Authentication', desc: 'Add extra security to your account', icon: '🛡️', color: 'green' },
                    { title: 'Download Data', desc: 'Export all your health data', icon: '📥', color: 'gray' },
                    { title: 'Delete Account', desc: 'Permanently delete your account and data', icon: '🗑️', color: 'red' }
                  ].map((action, index) => (
                    <button 
                      key={index}
                      className={`w-full text-left p-4 border rounded-lg hover:bg-gray-50 transition-colors ${
                        action.color === 'red' ? 'border-red-300 hover:bg-red-50' : 'border-gray-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <span className="text-xl">{action.icon}</span>
                          <div>
                            <h4 className={`text-sm font-medium ${action.color === 'red' ? 'text-red-900' : 'text-gray-900'}`}>
                              {action.title}
                            </h4>
                            <p className={`text-xs ${action.color === 'red' ? 'text-red-600' : 'text-gray-500'}`}>
                              {action.desc}
                            </p>
                          </div>
                        </div>
                        <svg className={`w-5 h-5 ${action.color === 'red' ? 'text-red-400' : 'text-gray-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;

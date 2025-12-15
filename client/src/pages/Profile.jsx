import React, { useState } from 'react';

const Profile = () => {
  const [isEditing, setIsEditing] = useState(false);
  const [activeTab, setActiveTab] = useState('personal');
  
  const [profile, setProfile] = useState({
    personalInfo: {
      name: 'John Doe',
      email: 'john.doe@example.com',
      phone: '+1 (555) 123-4567',
      dateOfBirth: '1990-05-15',
      gender: 'Male',
      address: '123 Health St, Medical City, MC 12345'
    },
    medicalInfo: {
      bloodType: 'O+',
      height: '175',
      weight: '70',
      allergies: ['Penicillin', 'Shellfish'],
      medications: ['Vitamin D', 'Omega-3'],
      conditions: ['Hypertension'],
      emergencyContact: {
        name: 'Jane Doe',
        relationship: 'Spouse',
        phone: '+1 (555) 987-6543'
      }
    }
  });

  const handleInputChange = (section, field, value) => {
    setProfile(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
  };

  const handleSave = () => {
    setIsEditing(false);
    // TODO: Save to backend
  };

  const tabs = [
    { id: 'personal', name: 'Personal Info', icon: '👤' },
    { id: 'medical', name: 'Medical Info', icon: '🏥' },
    { id: 'emergency', name: 'Emergency Contact', icon: '🚨' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Profile Settings</h1>
          <p className="mt-1 text-sm text-gray-600">
            Manage your personal information and health details
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          {isEditing ? (
            <>
              <button
                onClick={() => setIsEditing(false)}
                className="bg-white text-gray-700 border border-gray-300 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                💾 Save Changes
              </button>
            </>
          ) : (
            <button
              onClick={() => setIsEditing(true)}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              ✏️ Edit Profile
            </button>
          )}
        </div>
      </div>

      {/* Profile Card */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center space-x-6">
          <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center">
            <span className="text-white text-3xl font-bold">
              {profile.personalInfo.name.split(' ').map(n => n[0]).join('')}
            </span>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{profile.personalInfo.name}</h2>
            <p className="text-gray-600">{profile.personalInfo.email}</p>
            <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
              <span>📅 Member since January 2024</span>
              <span>•</span>
              <span>🔄 Last updated 2 days ago</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabbed Content */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100">
        {/* Tab Navigation */}
        <div className="border-b border-gray-100">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Personal Information Tab */}
          {activeTab === 'personal' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                  <input
                    type="text"
                    value={profile.personalInfo.name}
                    onChange={(e) => handleInputChange('personalInfo', 'name', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                  <input
                    type="email"
                    value={profile.personalInfo.email}
                    onChange={(e) => handleInputChange('personalInfo', 'email', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Phone Number</label>
                  <input
                    type="tel"
                    value={profile.personalInfo.phone}
                    onChange={(e) => handleInputChange('personalInfo', 'phone', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Date of Birth</label>
                  <input
                    type="date"
                    value={profile.personalInfo.dateOfBirth}
                    onChange={(e) => handleInputChange('personalInfo', 'dateOfBirth', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Gender</label>
                  <select
                    value={profile.personalInfo.gender}
                    onChange={(e) => handleInputChange('personalInfo', 'gender', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Address</label>
                <textarea
                  value={profile.personalInfo.address}
                  onChange={(e) => handleInputChange('personalInfo', 'address', e.target.value)}
                  disabled={!isEditing}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                />
              </div>
            </div>
          )}

          {/* Medical Information Tab */}
          {activeTab === 'medical' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Blood Type</label>
                  <select
                    value={profile.medicalInfo.bloodType}
                    onChange={(e) => handleInputChange('medicalInfo', 'bloodType', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  >
                    {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'].map(type => (
                      <option key={type} value={type}>{type}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Height (cm)</label>
                  <input
                    type="number"
                    value={profile.medicalInfo.height}
                    onChange={(e) => handleInputChange('medicalInfo', 'height', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Weight (kg)</label>
                  <input
                    type="number"
                    value={profile.medicalInfo.weight}
                    onChange={(e) => handleInputChange('medicalInfo', 'weight', e.target.value)}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Allergies</label>
                  <div className="space-y-2">
                    {profile.medicalInfo.allergies.map((allergy, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <span className="px-2 py-1 bg-red-50 text-red-700 text-sm rounded-full border border-red-200">
                          {allergy}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Current Medications</label>
                  <div className="space-y-2">
                    {profile.medicalInfo.medications.map((medication, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <span className="px-2 py-1 bg-blue-50 text-blue-700 text-sm rounded-full border border-blue-200">
                          {medication}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Medical Conditions</label>
                  <div className="space-y-2">
                    {profile.medicalInfo.conditions.map((condition, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <span className="px-2 py-1 bg-yellow-50 text-yellow-700 text-sm rounded-full border border-yellow-200">
                          {condition}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Emergency Contact Tab */}
          {activeTab === 'emergency' && (
            <div className="space-y-6">
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-red-600 text-lg">🚨</span>
                  <h3 className="text-lg font-semibold text-red-800">Emergency Contact Information</h3>
                </div>
                <p className="text-red-700 text-sm">
                  This information will be used in case of medical emergencies. Please ensure it's always up to date.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Contact Name</label>
                  <input
                    type="text"
                    value={profile.medicalInfo.emergencyContact.name}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Relationship</label>
                  <input
                    type="text"
                    value={profile.medicalInfo.emergencyContact.relationship}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Phone Number</label>
                  <input
                    type="tel"
                    value={profile.medicalInfo.emergencyContact.phone}
                    disabled={!isEditing}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Profile;

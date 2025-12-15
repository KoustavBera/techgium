import React, { useState } from 'react';

const History = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('All');

  const assessments = [
    { 
      id: 1, 
      date: '2024-01-15', 
      type: 'General Health Assessment', 
      status: 'Completed', 
      score: 85,
      duration: '15 minutes',
      symptoms: ['Headache', 'Fatigue']
    },
    { 
      id: 2, 
      date: '2024-01-10', 
      type: 'Eye Examination', 
      status: 'Completed', 
      score: 92,
      duration: '12 minutes',
      symptoms: ['Eye strain', 'Blurred vision']
    },
    { 
      id: 3, 
      date: '2024-01-05', 
      type: 'Skin Analysis', 
      status: 'Pending Review', 
      score: null,
      duration: '8 minutes',
      symptoms: ['Rash', 'Itching']
    }
  ];

  const filteredAssessments = assessments.filter(assessment => {
    const matchesSearch = assessment.type.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'All' || assessment.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  const getStatusBadge = (status) => {
    const styles = {
      'Completed': 'bg-green-100 text-green-800',
      'Pending Review': 'bg-yellow-100 text-yellow-800',
      'In Progress': 'bg-blue-100 text-blue-800'
    };
    return styles[status] || 'bg-gray-100 text-gray-800';
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Assessment History</h1>
          <p className="mt-1 text-sm text-gray-600">
            Track your health journey and view past assessments
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <button className="bg-white text-gray-700 border border-gray-300 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
            📊 Export Data
          </button>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            📋 New Assessment
          </button>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0 sm:space-x-4">
          <div className="flex-1 max-w-md">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                placeholder="Search assessments..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Filter:</label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="All">All Status</option>
              <option value="Completed">Completed</option>
              <option value="Pending Review">Pending Review</option>
              <option value="In Progress">In Progress</option>
            </select>
          </div>
        </div>
      </div>

      {/* Assessment Cards */}
      <div className="space-y-4">
        {filteredAssessments.length === 0 ? (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-12 text-center">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">📋</span>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No assessments found</h3>
            <p className="text-gray-600 mb-6">
              {searchTerm || filterStatus !== 'All' 
                ? 'Try adjusting your search or filter criteria.'
                : 'Start your first health assessment to see your history here.'
              }
            </p>
            <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
              Start Assessment
            </button>
          </div>
        ) : (
          filteredAssessments.map((assessment) => (
            <div key={assessment.id} className="bg-white rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-3">
                      <h3 className="text-lg font-semibold text-gray-900">{assessment.type}</h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(assessment.status)}`}>
                        {assessment.status}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-6 text-sm text-gray-500 mb-4">
                      <div className="flex items-center space-x-1">
                        <span>📅</span>
                        <span>{new Date(assessment.date).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <span>⏱️</span>
                        <span>{assessment.duration}</span>
                      </div>
                    </div>

                    <div>
                      <p className="text-sm text-gray-600 mb-2">Symptoms assessed:</p>
                      <div className="flex flex-wrap gap-2">
                        {assessment.symptoms.map((symptom, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded-full border border-blue-200"
                          >
                            {symptom}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col items-end space-y-3 ml-6">
                    {assessment.score && (
                      <div className="text-center">
                        <p className="text-sm text-gray-500">Health Score</p>
                        <p className={`text-3xl font-bold ${getScoreColor(assessment.score)}`}>
                          {assessment.score}
                        </p>
                        <p className="text-sm text-gray-400">/100</p>
                      </div>
                    )}
                    
                    <div className="flex space-x-2">
                      <button className="px-3 py-1 text-sm text-blue-600 border border-blue-600 rounded-md hover:bg-blue-50 transition-colors">
                        View Details
                      </button>
                      {assessment.status === 'Completed' && (
                        <button className="px-3 py-1 text-sm text-green-600 border border-green-600 rounded-md hover:bg-green-50 transition-colors">
                          Download
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Summary Stats */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Assessment Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{assessments.length}</p>
            <p className="text-sm text-gray-600">Total Assessments</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {assessments.filter(a => a.status === 'Completed').length}
            </p>
            <p className="text-sm text-gray-600">Completed</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-yellow-600">
              {assessments.filter(a => a.score && a.score >= 80).length}
            </p>
            <p className="text-sm text-gray-600">High Scores (80+)</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">
              {Math.round(assessments.filter(a => a.score).reduce((acc, a) => acc + a.score, 0) / assessments.filter(a => a.score).length) || 0}
            </p>
            <p className="text-sm text-gray-600">Average Score</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default History;

import React, { useState } from 'react';

const Reports = () => {
  const [selectedReport, setSelectedReport] = useState(null);

  const reports = [
    {
      id: 1,
      title: 'Comprehensive Health Analysis',
      date: '2024-01-15',
      type: 'Full Report',
      status: 'Ready',
      size: '2.4 MB',
      pages: 12,
      insights: [
        'Overall health score: 85/100',
        'Cardiovascular health: Good',
        'Recommended follow-up in 3 months'
      ]
    },
    {
      id: 2,
      title: 'Eye Health Assessment',
      date: '2024-01-10',
      type: 'Specialized Report',
      status: 'Ready',
      size: '1.8 MB',
      pages: 8,
      insights: [
        'Vision clarity: 20/20',
        'No signs of eye strain',
        'Recommended screen time breaks'
      ]
    },
    {
      id: 3,
      title: 'Lab Results Analysis',
      date: '2024-01-05',
      type: 'Lab Report',
      status: 'Processing',
      size: null,
      pages: null,
      insights: []
    }
  ];

  const getStatusBadge = (status) => {
    const styles = {
      'Ready': 'bg-green-100 text-green-800',
      'Processing': 'bg-yellow-100 text-yellow-800',
      'Failed': 'bg-red-100 text-red-800'
    };
    return styles[status] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Health Reports</h1>
          <p className="mt-1 text-sm text-gray-600">
            Download and review your AI-generated health analysis reports
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <button className="bg-white text-gray-700 border border-gray-300 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
            📋 Request Report
          </button>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            📊 Generate Summary
          </button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <span className="text-2xl">📄</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Reports</p>
              <p className="text-2xl font-bold text-gray-900">{reports.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <span className="text-2xl">✅</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Ready to Download</p>
              <p className="text-2xl font-bold text-gray-900">
                {reports.filter(r => r.status === 'Ready').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                <span className="text-2xl">⏳</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Processing</p>
              <p className="text-2xl font-bold text-gray-900">
                {reports.filter(r => r.status === 'Processing').length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Reports List */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900">Available Reports</h3>
        </div>
        <div className="divide-y divide-gray-100">
          {reports.map((report) => (
            <div key={report.id} className="p-6 hover:bg-gray-50 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-3">
                    <h4 className="text-lg font-semibold text-gray-900">{report.title}</h4>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(report.status)}`}>
                      {report.status}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-6 text-sm text-gray-500 mb-4">
                    <div className="flex items-center space-x-1">
                      <span>📅</span>
                      <span>{new Date(report.date).toLocaleDateString()}</span>
                    </div>
                    <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                      {report.type}
                    </span>
                    {report.size && (
                      <div className="flex items-center space-x-1">
                        <span>📁</span>
                        <span>{report.size} • {report.pages} pages</span>
                      </div>
                    )}
                  </div>

                  {report.insights.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-gray-700 mb-2">Key Insights:</p>
                      <ul className="space-y-1">
                        {report.insights.map((insight, index) => (
                          <li key={index} className="text-sm text-gray-600 flex items-start">
                            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                            {insight}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <div className="flex flex-col space-y-2 ml-6">
                  {report.status === 'Ready' ? (
                    <>
                      <button className="flex items-center px-4 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors">
                        <span className="mr-2">📥</span>
                        Download
                      </button>
                      <button 
                        onClick={() => setSelectedReport(report)}
                        className="flex items-center px-4 py-2 text-sm text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <span className="mr-2">👁️</span>
                        Preview
                      </button>
                      <button className="flex items-center px-4 py-2 text-sm text-green-600 border border-green-300 rounded-lg hover:bg-green-50 transition-colors">
                        <span className="mr-2">📤</span>
                        Share
                      </button>
                    </>
                  ) : (
                    <div className="flex items-center px-4 py-2 text-sm text-gray-500">
                      <div className="animate-spin w-4 h-4 mr-2 border-2 border-gray-300 border-t-blue-600 rounded-full"></div>
                      Processing...
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Report Preview Modal */}
      {selectedReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-100 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Report Preview</h3>
              <button
                onClick={() => setSelectedReport(null)}
                className="text-gray-400 hover:text-gray-600 p-1"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6">
              <h4 className="text-xl font-bold text-gray-900 mb-4">{selectedReport.title}</h4>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <p className="text-blue-800 text-sm">
                  📋 This is a preview of your health report. The full report contains detailed analysis,
                  recommendations, and visual charts based on your assessment data.
                </p>
              </div>
              <div className="space-y-4">
                {selectedReport.insights.map((insight, index) => (
                  <div key={index} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <p className="text-sm text-gray-700">{insight}</p>
                  </div>
                ))}
              </div>
              <div className="mt-6 flex space-x-3">
                <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                  Download Full Report
                </button>
                <button 
                  onClick={() => setSelectedReport(null)}
                  className="bg-gray-100 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  Close Preview
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Reports;

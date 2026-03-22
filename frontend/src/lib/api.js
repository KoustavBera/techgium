/**
 * lib/api.js
 * Centralised API configuration and fetch helpers for
 * the Chiranjeevi v3 frontend.
 */

export const API_BASE = 'http://localhost:8000';

/**
 * Perform a standard JSON fetch.
 * @param {string} path - Relative path, e.g. '/health'
 * @param {RequestInit} [options]
 * @returns {Promise<any>} Parsed JSON body
 */
export async function apiFetch(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            const err = await response.json();
            detail = err.detail || err.message || detail;
        } catch { /* ignore parse error */ }
        throw new Error(detail);
    }
    return response.json();
}

/**
 * Check backend health — resolves to true/false.
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        return res.ok;
    } catch {
        return false;
    }
}

/**
 * Open a streaming fetch for SSE-style endpoints.
 * Returns the raw Response so callers can read the body stream.
 * @param {string} path
 * @param {object} body  JSON body
 * @returns {Promise<Response>}
 */
export async function streamFetch(path, body, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        ...options,
    });
    if (!response.ok) {
        throw new Error(`Stream request failed: HTTP ${response.status}`);
    }
    return response;
}

// ── Hardware / Screening endpoints ──────────────────────────────

/**
 * Fetch current sensor connection status.
 * @returns {Promise<{camera: object, esp32: object, radar: object}>}
 */
export const getSensorStatus = () =>
    apiFetch('/api/v1/hardware/sensor-status');

/**
 * Start a new screening scan.
 * @param {object} params
 * @returns {Promise<object>}
 */
export const startScreening = (params) =>
    apiFetch('/api/v1/hardware/start-screening', {
        method: 'POST',
        body: JSON.stringify(params),
    });

/**
 * Poll current scan progress.
 * @returns {Promise<object>}
 */
export const getScanStatus = () =>
    apiFetch('/api/v1/hardware/scan-status');

/**
 * Get the report download URL.
 * @param {string} reportId
 * @returns {string}
 */
export const getReportDownloadUrl = (reportId) =>
    `${API_BASE}/api/v1/reports/${reportId}/download`;

/**
 * Get the QR code URL for mobile report download.
 * @param {string} reportId
 * @returns {string}
 */
export const getReportQrUrl = (reportId) =>
    `${API_BASE}/api/v1/reports/${reportId}/qr?t=${Date.now()}`;

/**
 * Get the camera MJPEG video feed URL.
 * @returns {string}
 */
export const getVideoFeedUrl = () =>
    `${API_BASE}/api/v1/hardware/video-feed?t=${Date.now()}`;

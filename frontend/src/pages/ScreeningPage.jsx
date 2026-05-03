/**
 * ScreeningPage.jsx
 * The main Health Screening dashboard integrating Camera, Controls, Progress, and Trust metrics.
 */
import { useState } from 'react'
import { API_BASE } from '../lib/api'
import { useScreening } from '../hooks/useScreening'
import CameraFeed from '../components/screening/CameraFeed'
import ScreeningControls from '../components/screening/ScreeningControls'
import ScanProgress from '../components/screening/ScanProgress'
import TrustDashboard from '../components/screening/TrustDashboard'

export default function ScreeningPage() {
    const [isCameraActive, setIsCameraActive] = useState(false)

    // API_BASE is imported from lib/api.js (reads VITE_API_BASE env var with localhost:8000 fallback)

    const {
        scanState,
        phase,
        activity,
        message,
        progress,
        userWarnings,
        uiHint,
        phaseIcon,
        stableFramesPct,
        trustMetadata,
        reportId,
        sensorStatus,
        startScan,
        checkSensors
    } = useScreening()

    const handleToggleCamera = () => {
        setIsCameraActive(prev => !prev)
    }

    const handleStartScreening = (config) => {
        // Force camera on when starting a scan
        setIsCameraActive(true)
        startScan(config)
    }

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'row',
            gap: '16px',
            height: '100%',
            // Break to column on smaller screens
            flexWrap: 'wrap',
            overflowY: 'auto',
            paddingBottom: '16px' // For nice scroll clearance
        }}>

            {/* ── Left Column: Camera Feed ── */}
            <div style={{
                flex: '1 1 60%',
                minWidth: '400px',
                display: 'flex',
                flexDirection: 'column',
                alignSelf: 'flex-start',
                position: 'sticky',
                top: 0,
                zIndex: 10
            }}>
                <CameraFeed
                    isActive={isCameraActive}
                    apiBase={API_BASE}
                    userWarnings={userWarnings}
                    scanState={scanState}
                    phase={phase}
                    activity={activity}
                    uiHint={uiHint}
                    phaseIcon={phaseIcon}
                />
            </div>

            {/* ── Right Column: Controls & Dynamic Dashboards ── */}
            <div style={{
                flex: '1 1 35%',
                minWidth: '340px',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px'
            }}>
                {/* Only show config controls if idle or complete (or errors out) */}
                {scanState !== 'running' && scanState !== 'initializing' && scanState !== 'complete' && (
                    <ScreeningControls
                        sensorStatus={sensorStatus}
                        onCheckSensors={checkSensors}
                        onStartScreening={handleStartScreening}
                        scanState={scanState}
                        isCameraActive={isCameraActive}
                        onToggleCamera={handleToggleCamera}
                    />
                )}

                {/* Show live progress during scan or immediately after */}
                {scanState !== 'idle' && (
                    <ScanProgress
                        phase={phase}
                        message={message}
                        scanState={scanState}
                        progress={progress}
                        uiHint={uiHint}
                        phaseIcon={phaseIcon}
                        stableFramesPct={stableFramesPct}
                    />
                )}

                {/* Show Trust Dashboard / Report Download when it completes */}
                {scanState === 'complete' && (
                    <TrustDashboard
                        scanState={scanState}
                        reportId={reportId}
                        trustMetadata={trustMetadata}
                        apiBase={API_BASE}
                    />
                )}

                {/* Complete State Reset Button */}
                {(scanState === 'complete' || scanState === 'error') && (
                    <button
                        onClick={() => window.location.reload()}
                        style={{
                            marginTop: 'auto', padding: '12px',
                            background: 'transparent',
                            color: 'var(--md-primary)', border: '1px solid var(--md-outline)',
                            borderRadius: '999px', cursor: 'pointer',
                            fontFamily: "'Google Sans', sans-serif", fontWeight: 500
                        }}
                    >
                        Start New Patient Scan
                    </button>
                )}
            </div>

        </div>
    )
}

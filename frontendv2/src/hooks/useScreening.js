/**
 * useScreening.js
 * Hook to manage the Health Screening state machine and hardware polling.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import * as API from '../lib/api' // using the standard fetch wrapper

const POLL_INTERVAL_MS = 500

export function useScreening() {
    // Overall state: idle | initializing | running | complete | error
    const [scanState, setScanState] = useState('idle')
    // Combined poll payload — single setState per tick avoids cascading renders
    const [scanStatus, setScanStatus] = useState({
        phase: 'IDLE',
        message: 'SYSTEM READY',
        progress: 0,
        userWarnings: null,
    })

    // Convenience aliases (keep API surface the same for consumers)
    const phase = scanStatus.phase
    const message = scanStatus.message
    const progress = scanStatus.progress
    const userWarnings = scanStatus.userWarnings
    const [trustMetadata, setTrustMetadata] = useState(null)

    // Results
    const [reportId, setReportId] = useState(null)
    const [errorMsg, setErrorMsg] = useState(null)

    // Sensor Connections
    const [sensorStatus, setSensorStatus] = useState({
        camera: { status: 'unknown' },
        esp32: { status: 'unknown' },
        radar: { status: 'unknown' }
    })

    // Polling Refs
    const continuousPollRef = useRef(null)
    const scanPollRef = useRef(null)

    // 1. Initial Sensor Check
    const checkSensors = useCallback(async () => {
        try {
            const data = await API.getSensorStatus()
            setSensorStatus(data)
        } catch (e) {
            console.error('Failed to check sensors')
        }
    }, [])

    // 2. Start Scan
    const startScan = useCallback(async (config) => {
        setScanState('initializing')
        setScanStatus(prev => ({ ...prev, phase: 'INITIALIZING', message: 'Preparing sensors...', userWarnings: null }))
        setErrorMsg(null)
        setTrustMetadata(null)
        setReportId(null)

        // Stop the background "idle" polling so it doesn't conflict
        if (continuousPollRef.current) {
            clearInterval(continuousPollRef.current)
            continuousPollRef.current = null
        }

        try {
            const res = await fetch(`${import.meta.env.VITE_API_BASE || 'http://localhost:8000'}/api/v1/hardware/start-screening`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    patient_id: config.patientId,
                    radar_port: config.radarPort,
                    esp32_port: config.esp32Port,
                    camera_index: 0,
                    age: config.age,
                    gender: config.gender,
                    activity_mode: config.activityMode
                })
            })

            if (!res.ok) {
                const errorData = await res.json()
                throw new Error(errorData.detail || `Server error: ${res.status}`)
            }

            const data = await res.json()
            if (data.status === 'error') {
                throw new Error(data.error || 'Failed to start scan')
            }

            setScanState('running')

            // Begin active polling
            if (!scanPollRef.current) {
                scanPollRef.current = setInterval(pollScanStatus, POLL_INTERVAL_MS)
            }

        } catch (e) {
            setScanState('error')
            setScanStatus(prev => ({ ...prev, phase: 'ERROR', message: e.message }))
            setErrorMsg(e.message)
            startContinuousPolling() // Resume idle polling
        }
    }, [])

    // 4. Continuous Idle Polling (For distance warnings pre-scan)
    const startContinuousPolling = useCallback(() => {
        if (continuousPollRef.current) clearInterval(continuousPollRef.current)

        continuousPollRef.current = setInterval(async () => {
            try {
                const status = await API.getScanStatus()
                // Only update idle warnings if we aren't actively running a scan
                if (status.state === 'idle' || status.state === 'complete') {
                    if (status.user_warnings) {
                        setScanStatus(prev => ({ ...prev, userWarnings: status.user_warnings }))
                    } else {
                        setScanStatus(prev => ({ ...prev, userWarnings: null }))
                    }
                }
            } catch {
                // Silent fail for background check
            }
        }, POLL_INTERVAL_MS)
    }, [])

    // 3. Poll Scan Status (Active Scan)
    const pollScanStatus = useCallback(async () => {
        try {
            const status = await API.getScanStatus()

            setScanStatus({
                phase: status.phase,
                message: status.message,
                progress: status.progress ?? 0,
                userWarnings: status.user_warnings ?? null,
            })

            // Terminal States
            if (status.state === 'complete') {
                if (scanPollRef.current) clearInterval(scanPollRef.current)
                scanPollRef.current = null

                setScanState('complete')
                setReportId(status.patient_report_id)
                if (status.trust_metadata) {
                    setTrustMetadata(status.trust_metadata)
                }

                startContinuousPolling() // Resume idle polling for next person
            } else if (status.state === 'error') {
                if (scanPollRef.current) clearInterval(scanPollRef.current)
                scanPollRef.current = null

                setScanState('error')
                setErrorMsg(status.message)
                setScanStatus(prev => ({ ...prev, phase: 'ERROR', message: status.message }))

                startContinuousPolling()
            }
        } catch (e) {
            console.error('Error polling scan status:', e)
        }
    }, [startContinuousPolling])

    // 5. Reset
    const resetScreening = useCallback(() => {
        setScanState('idle')
        setScanStatus({ phase: 'IDLE', message: 'SYSTEM READY', progress: 0, userWarnings: null })
    }, [])

    // Lifecycle
    useEffect(() => {
        checkSensors()
        startContinuousPolling()

        return () => {
            if (continuousPollRef.current) clearInterval(continuousPollRef.current)
            if (scanPollRef.current) clearInterval(scanPollRef.current)
        }
    }, [checkSensors, startContinuousPolling])

    return {
        scanState,
        phase,
        message,
        progress,
        userWarnings,
        trustMetadata,
        reportId,
        errorMsg,
        sensorStatus,
        startScan,
        resetScreening,
        checkSensors
    }
}

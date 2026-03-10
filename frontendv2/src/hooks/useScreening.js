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
    // Granular phase: IDLE, INITIALIZING, FACE_AND_VITALS, BODY_ANALYSIS, PROCESSING, COMPLETE, ERROR
    const [phase, setPhase] = useState('IDLE')
    const [message, setMessage] = useState('SYSTEM READY')
    const [progress, setProgress] = useState(0)

    // Warnings / Trust Data
    const [userWarnings, setUserWarnings] = useState(null)
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
        setPhase('INITIALIZING')
        setMessage('Preparing sensors...')
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
            setPhase('ERROR')
            setErrorMsg(e.message)
            setMessage(e.message)
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
                        setUserWarnings(status.user_warnings)
                    } else {
                        setUserWarnings(null)
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

            setPhase(status.phase)
            setMessage(status.message)
            setProgress(status.progress)

            if (status.user_warnings) {
                setUserWarnings(status.user_warnings)
            }

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

                startContinuousPolling()
            }
        } catch (e) {
            console.error('Error polling scan status:', e)
        }
    }, [startContinuousPolling])

    // 5. Reset
    const resetScreening = useCallback(() => {
        setScanState('idle')
        setPhase('IDLE')
        setMessage('SYSTEM READY')
        setProgress(0)
        setReportId(null)
        setTrustMetadata(null)
        setErrorMsg(null)
        setUserWarnings(null)
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

/**
 * RoomCalibrationButton.jsx
 *
 * Standalone "Calibrate Room" button + inline status panel for the
 * System Controls card. Calls POST /api/v1/hardware/calibrate and
 * polls GET /api/v1/hardware/calibrate/status until complete.
 *
 * Shows:
 *   - Last calibration timestamp (if any)
 *   - Animated progress bar while running (~8s)
 *   - Room temp, offset, lighting_ok, DHT11 source when complete
 *   - Warning if no calibration has been done yet
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ThermostatRoundedIcon from '@mui/icons-material/ThermostatRounded'
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded'
import WarningAmberRoundedIcon from '@mui/icons-material/WarningAmberRounded'
import ErrorRoundedIcon from '@mui/icons-material/ErrorRounded'
import { startRoomCalibration, getRoomCalibrationStatus } from '../../lib/api'

const POLL_MS = 500  // poll every 500 ms while calibration is running

/* ── Small label-value row ─────────────────────────────────────── */
function InfoRow({ label, value, ok }) {
    const color = ok === undefined
        ? 'var(--md-on-surface)'
        : ok ? '#4caf50' : '#f44336'
    return (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--md-on-surface-variant)' }}>{label}</span>
            <span style={{ fontSize: '11px', fontWeight: 600, color }}>{value}</span>
        </div>
    )
}

/* ── Animated progress bar ─────────────────────────────────────── */
function ProgressBar({ pct }) {
    return (
        <div style={{
            height: '4px', borderRadius: '4px',
            background: 'rgba(0,0,0,0.1)', overflow: 'hidden',
        }}>
            <motion.div
                animate={{ width: `${pct}%` }}
                transition={{ ease: 'easeOut', duration: 0.3 }}
                style={{ height: '100%', background: 'var(--md-primary)', borderRadius: '4px' }}
            />
        </div>
    )
}

/* ── Format ISO timestamp to readable string ───────────────────── */
function fmtTs(iso) {
    if (!iso) return null
    try {
        const d = new Date(iso)
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            + ' · ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' })
    } catch { return iso }
}

/* ── Main component ────────────────────────────────────────────── */
export default function RoomCalibrationButton({ disabled = false }) {
    // 'idle' | 'running' | 'complete' | 'error'
    const [calState, setCalState] = useState('idle')
    const [progress, setProgress] = useState(0)
    const [message, setMessage] = useState('')
    const [envCal, setEnvCal] = useState(null)      // env_calibration from status
    const [calibratedAt, setCalibratedAt] = useState(null)
    const [showPanel, setShowPanel] = useState(false)
    const pollTimer = useRef(null)

    /* ── Fetch latest status and apply it ─── */
    const fetchStatus = useCallback(async () => {
        try {
            const data = await getRoomCalibrationStatus()
            const state = data.state ?? 'idle'
            setCalState(state)
            setProgress(data.progress ?? 0)
            setMessage(data.message ?? '')
            if (data.env_calibration) setEnvCal(data.env_calibration)
            if (data.calibrated_at)   setCalibratedAt(data.calibrated_at)
            return state
        } catch {
            return calState
        }
    }, [calState])

    /* ── Load existing status on mount ─────── */
    useEffect(() => {
        fetchStatus()
    }, []) // eslint-disable-line react-hooks/exhaustive-deps

    /* ── Poll while running ─────────────────── */
    useEffect(() => {
        if (calState === 'running') {
            pollTimer.current = setInterval(async () => {
                const st = await fetchStatus()
                if (st !== 'running') clearInterval(pollTimer.current)
            }, POLL_MS)
        }
        return () => clearInterval(pollTimer.current)
    }, [calState, fetchStatus])

    /* ── Trigger calibration ────────────────── */
    const handleCalibrate = async () => {
        if (disabled || calState === 'running') return
        setShowPanel(true)
        setCalState('running')
        setProgress(0)
        setMessage('Starting calibration…')
        try {
            await startRoomCalibration()
        } catch (err) {
            // 409 means already running — just start polling
            if (!String(err.message).includes('409')) {
                setCalState('error')
                setMessage(String(err.message))
            }
        }
    }

    /* ── Button style based on state ───────── */
    const btnBg = calState === 'complete'
        ? 'var(--md-success-container, #c8e6c9)'
        : calState === 'error'
            ? 'var(--md-error-container)'
            : calState === 'running'
                ? 'var(--md-secondary-container)'
                : 'var(--md-surface-container-high)'

    const btnColor = calState === 'complete'
        ? 'var(--md-on-success-container, #1b5e20)'
        : calState === 'error'
            ? 'var(--md-on-error-container)'
            : 'var(--md-on-surface-variant)'

    const isRunning = calState === 'running'

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>

            {/* ── Trigger Button ─── */}
            <motion.button
                id="btn-room-calibrate"
                whileHover={!isRunning && !disabled ? { scale: 1.02 } : {}}
                whileTap={!isRunning && !disabled ? { scale: 0.97 } : {}}
                onClick={handleCalibrate}
                disabled={isRunning || disabled}
                aria-label="Calibrate room environment"
                style={{
                    height: '48px',
                    padding: '0 20px',
                    background: btnBg,
                    color: btnColor,
                    border: 'none',
                    borderRadius: '999px',
                    fontFamily: "'Google Sans', sans-serif",
                    fontSize: '14px',
                    fontWeight: 500,
                    cursor: isRunning || disabled ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    opacity: disabled ? 0.5 : 1,
                    transition: 'background 0.25s, color 0.25s',
                }}
            >
                {/* Spinning icon while running */}
                <motion.span
                    animate={isRunning ? { rotate: 360 } : { rotate: 0 }}
                    transition={isRunning ? { repeat: Infinity, duration: 2, ease: 'linear' } : {}}
                    style={{ display: 'flex', alignItems: 'center' }}
                >
                    <ThermostatRoundedIcon style={{ fontSize: 20 }} />
                </motion.span>

                {isRunning
                    ? `Calibrating… ${progress}%`
                    : calState === 'complete'
                        ? '✓ Room Calibrated'
                        : calState === 'error'
                            ? '⚠ Calibration Failed — Retry'
                            : 'Calibrate Room'}
            </motion.button>

            {/* ── Compact last-calibrated timestamp (always visible if done) ── */}
            <AnimatePresence>
                {calibratedAt && calState !== 'running' && (
                    <motion.div
                        key="ts"
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '5px',
                            paddingLeft: '4px',
                        }}
                    >
                        <CheckCircleRoundedIcon style={{ fontSize: 13, color: '#4caf50' }} />
                        <span style={{
                            fontSize: '11px',
                            color: 'var(--md-on-surface-variant)',
                            fontFamily: "'Google Sans', sans-serif",
                        }}>
                            Last calibrated: <strong>{fmtTs(calibratedAt)}</strong>
                        </span>
                    </motion.div>
                )}
                {!calibratedAt && calState === 'idle' && (
                    <motion.div
                        key="nocal"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{
                            display: 'flex', alignItems: 'center', gap: '5px', paddingLeft: '4px'
                        }}
                    >
                        <WarningAmberRoundedIcon style={{ fontSize: 13, color: '#ff9800' }} />
                        <span style={{ fontSize: '11px', color: '#e65100', fontFamily: "'Google Sans', sans-serif" }}>
                            Not calibrated — using defaults (25°C, +0.8°C offset)
                        </span>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Expandable status panel ─── */}
            <AnimatePresence>
                {showPanel && (
                    <motion.div
                        key="panel"
                        initial={{ opacity: 0, height: 0, marginTop: 0 }}
                        animate={{ opacity: 1, height: 'auto', marginTop: 4 }}
                        exit={{ opacity: 0, height: 0, marginTop: 0 }}
                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                        style={{ overflow: 'hidden' }}
                    >
                        <div style={{
                            background: 'var(--md-surface-container)',
                            borderRadius: '16px',
                            padding: '14px 16px',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '10px',
                            border: '1px solid var(--md-outline-variant)',
                        }}>
                            {/* Header row */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{
                                    fontSize: '11px', fontWeight: 700, letterSpacing: '0.8px',
                                    textTransform: 'uppercase', color: 'var(--md-on-surface-variant)',
                                }}>
                                    🌡️ Room Calibration
                                </span>
                                <button
                                    onClick={() => setShowPanel(false)}
                                    style={{
                                        background: 'none', border: 'none', cursor: 'pointer',
                                        color: 'var(--md-on-surface-variant)', fontSize: '16px',
                                        lineHeight: 1, padding: '0 2px',
                                    }}
                                    aria-label="Close calibration panel"
                                >×</button>
                            </div>

                            {/* Progress bar (visible while running) */}
                            <AnimatePresence>
                                {isRunning && (
                                    <motion.div
                                        key="bar"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}
                                    >
                                        <ProgressBar pct={progress} />
                                        <span style={{
                                            fontSize: '12px',
                                            color: 'var(--md-on-surface-variant)',
                                            fontFamily: "'Google Sans', sans-serif",
                                        }}>
                                            {message || 'Measuring environment…'}
                                        </span>
                                        <span style={{ fontSize: '11px', color: 'var(--md-outline)' }}>
                                            ⚠ Keep the camera view clear — step aside
                                        </span>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {/* Results (visible when complete or error) */}
                            <AnimatePresence>
                                {calState === 'complete' && envCal && (
                                    <motion.div
                                        key="results"
                                        initial={{ opacity: 0, y: 4 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ duration: 0.3 }}
                                        style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}
                                    >
                                        <InfoRow
                                            label="Room Temperature"
                                            value={`${envCal.room_temp?.toFixed(1) ?? '--'}°C`}
                                        />
                                        <InfoRow
                                            label="Thermal Offset"
                                            value={`+${envCal.dynamic_temp_offset?.toFixed(2) ?? '--'}°C`}
                                        />
                                        <InfoRow
                                            label="Lighting"
                                            value={envCal.lighting_ok ? '✓ OK' : '⚠ Low'}
                                            ok={envCal.lighting_ok}
                                        />
                                        <InfoRow
                                            label="Temp Source"
                                            value={envCal.dht11_used ? 'DHT11 sensor ✓' : 'Thermal bg-pixel'}
                                            ok={envCal.dht11_used}
                                        />
                                        {!envCal.lighting_ok && (
                                            <div style={{
                                                marginTop: '4px', padding: '8px 10px',
                                                background: 'rgba(255,152,0,0.1)',
                                                borderRadius: '10px', fontSize: '11px',
                                                color: '#e65100', lineHeight: 1.5,
                                            }}>
                                                ⚠ Low lighting detected. For best accuracy, improve room
                                                illumination before scanning.
                                            </div>
                                        )}
                                        <div style={{
                                            display: 'flex', alignItems: 'center', gap: '6px',
                                            marginTop: '2px', padding: '8px 10px',
                                            background: 'rgba(76,175,80,0.08)', borderRadius: '10px',
                                        }}>
                                            <CheckCircleRoundedIcon style={{ fontSize: 15, color: '#4caf50' }} />
                                            <span style={{ fontSize: '11px', color: '#2e7d32', fontWeight: 500 }}>
                                                Calibration saved — valid for all scans until manually reset
                                            </span>
                                        </div>
                                    </motion.div>
                                )}

                                {calState === 'error' && (
                                    <motion.div
                                        key="err"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        style={{
                                            display: 'flex', alignItems: 'flex-start', gap: '8px',
                                            padding: '10px', background: 'rgba(244,67,54,0.08)',
                                            borderRadius: '10px',
                                        }}
                                    >
                                        <ErrorRoundedIcon style={{ fontSize: 16, color: '#f44336', flexShrink: 0, marginTop: '1px' }} />
                                        <span style={{ fontSize: '12px', color: '#c62828', lineHeight: 1.5 }}>
                                            {message || 'Calibration failed. Default values will be used.'}
                                        </span>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

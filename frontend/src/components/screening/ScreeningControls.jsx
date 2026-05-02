/**
 * ScreeningControls.jsx
 * Config form, sensor connection checks, and Patient Context modal.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import PhotoCameraRoundedIcon from '@mui/icons-material/PhotoCameraRounded'
import ThermostatRoundedIcon from '@mui/icons-material/ThermostatRounded'
import CellTowerRoundedIcon from '@mui/icons-material/CellTowerRounded'
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded'
import HighlightOffRoundedIcon from '@mui/icons-material/HighlightOffRounded'
import CalibrationPanel from './CalibrationPanel'

const cardStyle = {
    background: '#F3F4F4',
    borderRadius: '28px',
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
}

const inputStyle = {
    width: '100%',
    padding: '12px 16px',
    background: 'var(--md-surface-container-high)',
    border: 'none',
    borderBottom: '1px solid var(--md-outline)',
    borderRadius: '12px 12px 0 0',
    fontFamily: "'Google Sans', sans-serif",
    fontSize: '16px',
    color: 'var(--md-on-surface)',
    outline: 'none',
    transition: 'background 0.2s',
}

const labelStyle = {
    fontSize: '12px',
    fontWeight: 500,
    color: 'var(--md-on-surface-variant)',
    marginBottom: '4px',
    display: 'block',
    marginLeft: '4px'
}

function SensorStatus({ icon, name, status }) {
    const isConnected = status === 'connected'
    return (
        <div style={{
            background: isConnected ? 'var(--md-success-container)' : 'var(--md-surface)',
            color: isConnected ? 'var(--md-on-success-container)' : 'var(--md-on-surface-variant)',
            padding: '16px 8px',
            borderRadius: '16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '8px',
            flex: 1,
            transition: 'all 0.3s ease'
        }}>
            {icon}
            <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                {name}
            </div>
            <div style={{ fontSize: '12px', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '4px' }}>
                {isConnected ? <CheckCircleRoundedIcon style={{ fontSize: 16 }} /> : <HighlightOffRoundedIcon style={{ fontSize: 16 }} />}
                {isConnected ? 'Online' : 'Disc.'}
            </div>
        </div>
    )
}

export default function ScreeningControls({
    sensorStatus,
    onCheckSensors,
    onStartScreening,
    scanState, // 'idle', 'running', 'complete', 'error'
    isCameraActive,
    onToggleCamera
}) {
    const [config, setConfig] = useState({
        patientId: 'PATIENT_001',
        radarPort: 'COM6',
        esp32Port: 'COM7'
    })

    const [showModal, setShowModal] = useState(false)
    const [modalStep, setModalStep] = useState(1) // 1 = Patient Profile, 2 = Calibration
    const [context, setContext] = useState({
        age: '30',
        gender: 'male',
        activityMode: 'resting'
    })

    const handleStartSubmit = (e) => {
        e.preventDefault()
        setModalStep(1)
        setShowModal(true)
    }

    const handleProfileNext = () => setModalStep(2)
    const handleCalibrationBack = () => setModalStep(1)

    const handleConfirmStart = () => {
        setShowModal(false)
        setModalStep(1)
        onStartScreening({ ...config, ...context, age: parseInt(context.age) || 30 })
    }

    const isBusy = scanState === 'initializing' || scanState === 'running'

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* ── System Controls ── */}
            <div style={cardStyle}>
                <h2 style={{ fontSize: '24px', fontWeight: 400, margin: 0, fontFamily: "'Google Sans', sans-serif" }}>
                    System Controls
                </h2>

                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={onToggleCamera}
                    style={{
                        padding: '0 24px', height: '48px',
                        background: isCameraActive ? 'var(--md-error-container)' : 'var(--md-secondary-container)',
                        color: isCameraActive ? 'var(--md-on-error-container)' : 'var(--md-on-secondary-container)',
                        border: 'none', borderRadius: '999px',
                        fontFamily: "'Google Sans', sans-serif", fontSize: '14px', fontWeight: 500,
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'
                    }}
                >
                    {isCameraActive ? '⏹ Hide Camera' : '▶ Show Camera'}
                </motion.button>

                <div style={{ display: 'flex', gap: '8px' }}>
                    <SensorStatus icon={<PhotoCameraRoundedIcon />} name="RGB" status={sensorStatus?.camera?.status} />
                    <SensorStatus icon={<ThermostatRoundedIcon />} name="Thermal" status={sensorStatus?.esp32?.status} />
                    <SensorStatus icon={<CellTowerRoundedIcon />} name="Radar" status={sensorStatus?.radar?.status} />
                </div>

                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={onCheckSensors}
                    style={{
                        padding: '10px', background: 'transparent',
                        color: 'var(--md-primary)', border: '1px solid var(--md-outline)',
                        borderRadius: '999px', cursor: 'pointer',
                        fontFamily: "'Google Sans', sans-serif", fontSize: '14px', fontWeight: 500
                    }}
                >
                    Check Connectivity
                </motion.button>
            </div>

            {/* ── Configuration Form ── */}
            <div style={{ ...cardStyle, opacity: isBusy ? 0.5 : 1, pointerEvents: isBusy ? 'none' : 'auto' }}>
                <h2 style={{ fontSize: '24px', fontWeight: 400, margin: 0, fontFamily: "'Google Sans', sans-serif" }}>
                    Configuration
                </h2>
                <form onSubmit={handleStartSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div>
                        <label style={labelStyle}>Patient ID</label>
                        <input
                            style={inputStyle}
                            value={config.patientId}
                            onChange={e => setConfig({ ...config, patientId: e.target.value })}
                            required
                        />
                    </div>
                    <div style={{ display: 'flex', gap: '16px' }}>
                        <div style={{ flex: 1 }}>
                            <label style={labelStyle}>Radar Port</label>
                            <input
                                style={inputStyle}
                                value={config.radarPort}
                                onChange={e => setConfig({ ...config, radarPort: e.target.value })}
                            />
                        </div>
                        <div style={{ flex: 1 }}>
                            <label style={labelStyle}>Thermal Port</label>
                            <input
                                style={inputStyle}
                                value={config.esp32Port}
                                onChange={e => setConfig({ ...config, esp32Port: e.target.value })}
                            />
                        </div>
                    </div>
                    <motion.button
                        whileHover={{ scale: 1.02, boxShadow: '0 2px 6px rgba(0,0,0,0.2)' }}
                        whileTap={{ scale: 0.98 }}
                        type="submit"
                        style={{
                            height: '48px', marginTop: '8px',
                            background: 'var(--md-primary)', color: 'var(--md-on-primary)',
                            border: 'none', borderRadius: '999px',
                            fontFamily: "'Google Sans', sans-serif", fontSize: '14px', fontWeight: 500, letterSpacing: '0.25px',
                            cursor: 'pointer'
                        }}
                    >
                        Start Screening Flow
                    </motion.button>
                </form>
            </div>

            {/* ── Patient Context Modal (M3 Dialog) ── */}
            <AnimatePresence>
                {showModal && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{
                            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                            background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)',
                            zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}
                    >
                        <motion.div
                            className="scrollbar-none"
                            initial={{ scale: 0.9, y: 20 }}
                            animate={{ scale: 1, y: 0 }}
                            exit={{ scale: 0.9, y: 20 }}
                            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                            style={{
                                background: 'var(--md-surface)',
                                width: '90%', maxWidth: '400px',
                                padding: '24px', borderRadius: '28px',
                                boxShadow: '0 4px 24px rgba(0,0,0,0.15)',
                                maxHeight: '90vh',
                                overflowY: 'auto',
                            }}
                        >
                            {/* ── Step indicator ── */}
                            <div style={{ display: 'flex', gap: '6px', marginBottom: '20px' }}>
                                {[1, 2].map(s => (
                                    <div key={s} style={{
                                        flex: 1, height: '3px', borderRadius: '2px',
                                        background: s <= modalStep ? 'var(--md-primary)' : 'rgba(255,255,255,0.15)',
                                        transition: 'background 0.3s'
                                    }} />
                                ))}
                            </div>

                            <AnimatePresence mode="wait">
                            {modalStep === 1 && (
                                <motion.div
                                    key="step1"
                                    initial={{ opacity: 0, x: -12 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -12 }}
                                    transition={{ duration: 0.22 }}
                                >
                                    <h2 style={{ margin: '0 0 4px 0', fontFamily: "'Google Sans', sans-serif", fontSize: '22px', fontWeight: 500 }}>
                                        Patient Profile
                                    </h2>
                                    <p style={{ margin: '0 0 20px 0', fontSize: '13px', color: 'var(--md-on-surface-variant)' }}>
                                        Step 1 of 2 · Calibrate physiological baselines
                                    </p>

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                        <div>
                                            <label style={{ ...labelStyle, color: 'var(--md-primary)', textTransform: 'uppercase' }}>Age</label>
                                            <input
                                                type="number" style={{ ...inputStyle, borderRadius: '8px', border: '1px solid var(--md-outline)' }}
                                                value={context.age} onChange={e => setContext({ ...context, age: e.target.value })}
                                                min={5} max={120}
                                            />
                                        </div>
                                        <div>
                                            <label style={{ ...labelStyle, color: 'var(--md-primary)', textTransform: 'uppercase' }}>Gender</label>
                                            <select
                                                style={{ ...inputStyle, borderRadius: '8px', border: '1px solid var(--md-outline)' }}
                                                value={context.gender} onChange={e => setContext({ ...context, gender: e.target.value })}
                                            >
                                                <option value="male">Male</option>
                                                <option value="female">Female</option>
                                                <option value="other">Other</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label style={{ ...labelStyle, color: 'var(--md-primary)', textTransform: 'uppercase' }}>Activity Level</label>
                                            <select
                                                style={{ ...inputStyle, borderRadius: '8px', border: '1px solid var(--md-outline)' }}
                                                value={context.activityMode} onChange={e => setContext({ ...context, activityMode: e.target.value })}
                                            >
                                                <option value="resting">Resting (seated &gt;5 mins)</option>
                                                <option value="post_exercise">Post-Exercise / Active</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', marginTop: '28px' }}>
                                        <motion.button whileTap={{ scale: 0.95 }} onClick={() => setShowModal(false)}
                                            style={{ background: 'transparent', color: 'var(--md-on-surface-variant)', border: '1px solid var(--md-outline)', padding: '10px 20px', borderRadius: '24px', fontWeight: 500, cursor: 'pointer', fontFamily: "'Google Sans', sans-serif" }}
                                        >
                                            Cancel
                                        </motion.button>
                                        <motion.button whileTap={{ scale: 0.95 }} onClick={handleProfileNext}
                                            style={{ background: 'var(--md-primary)', color: 'var(--md-on-primary)', border: 'none', padding: '10px 28px', borderRadius: '24px', fontWeight: 500, cursor: 'pointer', fontFamily: "'Google Sans', sans-serif" }}
                                        >
                                            Next: Calibration →
                                        </motion.button>
                                    </div>
                                </motion.div>
                            )}

                            {modalStep === 2 && (
                                <motion.div
                                    key="step2"
                                    initial={{ opacity: 0, x: 12 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 12 }}
                                    transition={{ duration: 0.22 }}
                                >
                                    <CalibrationPanel
                                        onProceed={handleConfirmStart}
                                        onBack={handleCalibrationBack}
                                    />
                                </motion.div>
                            )}
                            </AnimatePresence>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

        </div>
    )
}

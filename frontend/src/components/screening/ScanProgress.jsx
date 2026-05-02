/**
 * ScanProgress.jsx
 * Visualizes the screening steps with real-time ui_hint, phase_icon, and signal quality bar.
 */
import { motion, AnimatePresence } from 'framer-motion'
import CheckRoundedIcon from '@mui/icons-material/CheckRounded'

const STEPS = [
    { id: 'init',    label: 'Initializing',         backendPhases: ['INITIALIZING'] },
    { id: 'fadjust', label: 'Face Positioning',      backendPhases: ['FACE_ADJUST', 'FACE_ADJUST_EXTENDED'] },
    { id: 'capture', label: 'Face & Vitals Scan',    backendPhases: ['FACE_AND_VITALS'] },
    { id: 'badjust', label: 'Body Positioning',      backendPhases: ['BODY_ADJUST', 'BODY_ADJUST_EXTENDED'] },
    { id: 'pose',    label: 'Body Analysis',         backendPhases: ['BODY_ANALYSIS'] },
    { id: 'process', label: 'AI Processing',         backendPhases: ['PROCESSING'] },
]

// Signal quality bar: color depends on pct value
function SignalQualityBar({ pct }) {
    if (pct === null || pct === undefined) return null

    const clampedPct = Math.max(0, Math.min(100, pct))
    const color = clampedPct >= 70
        ? '#0F9D58'   // green
        : clampedPct >= 50
            ? '#FBBC04' // amber
            : '#D93025' // red

    const label = clampedPct >= 70 ? 'Excellent signal' : clampedPct >= 50 ? 'Fair signal' : 'Poor signal'

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--md-on-surface-variant)', letterSpacing: '0.2px' }}>
                    📶 Signal Quality — {label}
                </span>
                <span style={{ fontSize: 12, fontWeight: 700, color }}>{Math.round(clampedPct)}%</span>
            </div>
            <div style={{ height: 6, background: 'var(--md-surface-container-high)', borderRadius: 99, overflow: 'hidden' }}>
                <motion.div
                    animate={{ width: `${clampedPct}%` }}
                    transition={{ duration: 0.6, ease: 'easeOut' }}
                    style={{ height: '100%', background: color, borderRadius: 99 }}
                />
            </div>
        </div>
    )
}

const ADJUST_PHASES = ['FACE_ADJUST', 'FACE_ADJUST_EXTENDED', 'BODY_ADJUST', 'BODY_ADJUST_EXTENDED']
const CAPTURE_PHASES = ['FACE_AND_VITALS', 'BODY_ANALYSIS']

export default function ScanProgress({ phase, message, scanState, progress, uiHint, phaseIcon, stableFramesPct }) {
    if (scanState === 'idle') return null

    const activeIndex = STEPS.findIndex(s => s.backendPhases.includes(phase))
    const isComplete = scanState === 'complete' || phase === 'COMPLETE'
    const isError = scanState === 'error' || phase === 'ERROR'
    const isAdjust = ADJUST_PHASES.includes(phase)
    const isCapture = CAPTURE_PHASES.includes(phase)

    return (
        <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            style={{
                background: 'var(--md-surface-container)',
                borderRadius: '28px',
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
                overflow: 'hidden'
            }}
        >
            <h2 style={{ fontSize: '20px', fontWeight: 400, margin: 0, fontFamily: "'Google Sans', sans-serif" }}>
                Status
            </h2>

            {/* ── Step Indicators ── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {STEPS.map((step, index) => {
                    const isActive = index === activeIndex
                    const isDone = isComplete || (activeIndex > -1 && index < activeIndex)

                    const activeStyle = isActive ? {
                        background: 'var(--md-surface-container-highest)',
                        position: 'relative',
                        overflow: 'hidden',
                        color: 'var(--md-on-surface)',
                        boxShadow: '0 0 16px rgba(50, 150, 255, 0.2), inset 0 0 8px rgba(255, 255, 255, 0.4)',
                        border: '1px solid rgba(99, 210, 255, 0.3)'
                    } : {
                        background: isDone ? 'var(--md-surface)' : 'var(--md-surface)',
                        color: isDone ? 'var(--md-on-surface-variant)' : 'var(--md-on-surface-variant)'
                    }

                    return (
                        <motion.div
                            key={step.id}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '14px',
                                padding: '10px 14px', borderRadius: '14px',
                                fontSize: '13px', fontWeight: 500,
                                transition: 'background 0.3s ease, color 0.3s ease',
                                ...activeStyle
                            }}
                        >
                            {/* Shimmer overlay on active step */}
                            {isActive && (
                                <motion.div
                                    animate={{ opacity: [0.3, 0.8, 0.3], backgroundPosition: ['-100% 50%', '200% 50%'] }}
                                    transition={{ duration: 3, ease: 'easeInOut', repeat: Infinity }}
                                    style={{
                                        position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                                        background: 'radial-gradient(circle at center, rgba(255,255,255,0.7) 0%, rgba(99,210,255,0.3) 30%, transparent 60%)',
                                        backgroundSize: '300% 100%',
                                        backgroundRepeat: 'no-repeat',
                                        pointerEvents: 'none',
                                        mixBlendMode: 'overlay'
                                    }}
                                />
                            )}
                            <div style={{
                                width: 26, height: 26, borderRadius: '50%', flexShrink: 0,
                                background: isDone ? 'var(--md-success-container)' : isActive ? 'var(--md-primary)' : 'var(--md-surface-container-highest)',
                                color: isDone ? 'var(--md-on-success-container)' : isActive ? 'var(--md-on-primary)' : 'var(--md-on-surface-variant)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '12px', fontWeight: 700
                            }}>
                                {isDone ? <CheckRoundedIcon style={{ fontSize: 14 }} /> : (index + 1)}
                            </div>
                            <span style={{ zIndex: 1 }}>{step.label}</span>
                            {/* ADJUST badge — no data captured */}
                            {isActive && isAdjust && (
                                <span style={{
                                    marginLeft: 'auto', zIndex: 1,
                                    fontSize: 10, fontWeight: 700, letterSpacing: '0.4px',
                                    padding: '2px 8px', borderRadius: 999,
                                    background: 'rgba(99,210,255,0.15)',
                                    color: 'rgba(99,210,255,0.95)',
                                    border: '1px solid rgba(99,210,255,0.3)',
                                    textTransform: 'uppercase'
                                }}>
                                    NO CAPTURE
                                </span>
                            )}
                        </motion.div>
                    )
                })}
            </div>

            {/* ── Signal Quality Bar (capture phases only) ── */}
            <AnimatePresence>
                {isCapture && (
                    <motion.div
                        key="quality-bar"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        style={{ overflow: 'hidden' }}
                    >
                        <SignalQualityBar pct={stableFramesPct} />
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── ui_hint Banner (user-facing guidance from backend) ── */}
            <AnimatePresence mode="wait">
                {uiHint && (
                    <motion.div
                        key={uiHint}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        transition={{ duration: 0.3 }}
                        style={{
                            padding: '12px 16px', borderRadius: '16px',
                            background: isAdjust
                                ? 'rgba(99,210,255,0.08)'
                                : 'var(--md-secondary-container)',
                            color: isAdjust
                                ? 'rgba(99,210,255,0.95)'
                                : 'var(--md-on-secondary-container)',
                            border: isAdjust
                                ? '1px solid rgba(99,210,255,0.2)'
                                : '1px solid transparent',
                            fontSize: '13px', fontWeight: 500,
                            display: 'flex', alignItems: 'center', gap: '10px'
                        }}
                    >
                        <span style={{ fontSize: 18, lineHeight: 1 }}>{phaseIcon || '💡'}</span>
                        <span>{uiHint}</span>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Progress Message Banner ── */}
            <motion.div
                style={{
                    padding: '14px 16px', borderRadius: '16px',
                    background: isError
                        ? 'var(--md-error-container)'
                        : isComplete
                            ? 'var(--md-success-container)'
                            : 'var(--md-primary-container)',
                    color: isError
                        ? 'var(--md-on-error-container)'
                        : isComplete
                            ? 'var(--md-on-success-container)'
                            : 'var(--md-on-primary-container)',
                    fontSize: '13px', fontWeight: 500,
                    display: 'flex', alignItems: 'center', gap: '10px'
                }}
            >
                <span>{isError ? '❌' : isComplete ? '✅' : '⏳'}</span>
                <span style={{ flex: 1 }}>{message}</span>
                {!isComplete && !isError && progress > 0 && (
                    <span style={{ fontWeight: 700, flexShrink: 0 }}>{progress}%</span>
                )}
            </motion.div>
        </motion.div>
    )
}

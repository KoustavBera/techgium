/**
 * ScanProgress.jsx
 * Visualizes the screening steps (Initializing -> Face/Vitals -> Body -> AI Processing).
 */
import { motion } from 'framer-motion'
import CheckRoundedIcon from '@mui/icons-material/CheckRounded'
import HourglassEmptyRoundedIcon from '@mui/icons-material/HourglassEmptyRounded'

const STEPS = [
    { id: 'init', label: 'Initializing', backendPhase: 'INITIALIZING' },
    { id: 'capture', label: 'Face & Vitals Scan (30s)', backendPhase: 'FACE_AND_VITALS' },
    { id: 'pose', label: 'Body Analysis (10s)', backendPhase: 'BODY_ANALYSIS' },
    { id: 'process', label: 'AI Processing', backendPhase: 'PROCESSING' }
]

export default function ScanProgress({ phase, message, scanState, progress }) {
    // If we're not running or complete/error, hiding this is usually cleaner, 
    // but let's show it if it's active.
    if (scanState === 'idle') return null

    // Determine active index
    const activeIndex = STEPS.findIndex(s => s.backendPhase === phase)
    const isComplete = scanState === 'complete' || phase === 'COMPLETE'
    const isError = scanState === 'error' || phase === 'ERROR'

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

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {STEPS.map((step, index) => {
                    const isActive = index === activeIndex
                    const isDone = isComplete || (activeIndex > -1 && index < activeIndex)

                    let bg = 'var(--md-surface)'
                    let color = 'var(--md-on-surface-variant)'

                    if (isActive) {
                        bg = 'var(--md-surface-container-high)'
                        color = 'var(--md-on-surface)'
                    } else if (isDone) {
                        bg = 'var(--md-surface)'
                        color = 'var(--md-on-surface-variant)'
                    }

                    return (
                        <motion.div
                            layout
                            key={step.id}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '16px',
                                padding: '12px 16px', borderRadius: '16px',
                                background: bg, color: color,
                                fontSize: '14px', fontWeight: 500,
                                transition: 'all 0.3s ease'
                            }}
                        >
                            <div style={{
                                width: 28, height: 28, borderRadius: '50%',
                                background: isDone ? 'var(--md-success-container)' : isActive ? 'var(--md-primary)' : 'var(--md-surface-container-highest)',
                                color: isDone ? 'var(--md-on-success-container)' : isActive ? 'var(--md-on-primary)' : 'var(--md-on-surface-variant)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '13px', fontWeight: 700
                            }}>
                                {isDone ? <CheckRoundedIcon style={{ fontSize: 16 }} /> : (index + 1)}
                            </div>
                            {step.label}

                            {isActive && (
                                <motion.div
                                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                                    style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--md-primary)' }}
                                >
                                    <motion.div
                                        animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                                        style={{ display: 'flex' }}
                                    >
                                        <HourglassEmptyRoundedIcon style={{ fontSize: 16 }} />
                                    </motion.div>
                                </motion.div>
                            )}
                        </motion.div>
                    )
                })}
            </div>

            {/* Progress Message Banner */}
            <motion.div
                layout
                style={{
                    marginTop: '8px', padding: '16px', borderRadius: '16px',
                    background: isError ? 'var(--md-error-container)' : isComplete ? 'var(--md-success-container)' : 'var(--md-primary-container)',
                    color: isError ? 'var(--md-on-error-container)' : isComplete ? 'var(--md-on-success-container)' : 'var(--md-on-primary-container)',
                    fontSize: '14px', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '12px'
                }}
            >
                <span>{isError ? '❌' : isComplete ? '✅' : '⏳'}</span>
                <span style={{ flex: 1 }}>{message}</span>
                {!isComplete && !isError && progress > 0 && (
                    <span style={{ fontWeight: 700 }}>{progress}%</span>
                )}
            </motion.div>
        </motion.div>
    )
}

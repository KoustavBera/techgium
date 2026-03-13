/**
 * CameraFeed.jsx
 * Displays the live MJPEG stream from the backend.
 * 
 * Features:
 * - Distance / face / lighting warnings (from backend user_warnings)
 * - Static head-position guide overlay shown pre-scan (IDLE / INITIALIZING)
 * - Hides guide once the scan goes live to avoid visual noise
 */
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import WarningRoundedIcon from '@mui/icons-material/WarningRounded'
import PersonRoundedIcon from '@mui/icons-material/PersonRounded'
import LightbulbRoundedIcon from '@mui/icons-material/LightbulbRounded'
import PhotoCameraRoundedIcon from '@mui/icons-material/PhotoCameraRounded'

// ── Helper: static SVG head-guide overlay ────────────────────────────────────

function HeadGuide() {
    return (
        <motion.div
            key="head-guide"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            style={{
                position: 'absolute',
                top: 0, left: 0, right: 0, bottom: 0,
                zIndex: 25,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                pointerEvents: 'none',
            }}
        >
            {/* Realistic-scale head + shoulder guide — 65% of container width */}
            <svg
                viewBox="0 0 300 420"
                style={{ width: '65%', maxWidth: 420 }}
                xmlns="http://www.w3.org/2000/svg"
            >
                <defs>
                    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
                        <feGaussianBlur stdDeviation="5" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Shoulder silhouette */}
                <path
                    d="M 0 420 Q 0 375 70 360 Q 150 340 230 360 Q 300 375 300 420 Z"
                    fill="rgba(255,255,255,0.05)"
                    stroke="rgba(255,255,255,0.22)"
                    strokeWidth="2"
                    strokeDasharray="8 5"
                />

                {/* Neck */}
                <rect
                    x="127" y="337" width="46" height="38"
                    rx="8"
                    fill="rgba(255,255,255,0.05)"
                    stroke="rgba(255,255,255,0.18)"
                    strokeWidth="1.5"
                />

                {/* Head oval — main guide ring */}
                <ellipse
                    cx="150" cy="210" rx="118" ry="138"
                    fill="rgba(255,255,255,0.03)"
                    stroke="rgba(99,210,255,0.85)"
                    strokeWidth="2.5"
                    strokeDasharray="14 7"
                    filter="url(#glow)"
                >
                    <animate
                        attributeName="stroke-dashoffset"
                        from="0" to="-42"
                        dur="2.4s"
                        repeatCount="indefinite"
                    />
                </ellipse>

                {/* Corner bracket ticks */}
                {/* top-left */}
                <path d="M 40 125 L 40 107 L 58 107" stroke="rgba(99,210,255,0.9)" strokeWidth="2.5" fill="none" strokeLinecap="round" />
                {/* top-right */}
                <path d="M 260 125 L 260 107 L 242 107" stroke="rgba(99,210,255,0.9)" strokeWidth="2.5" fill="none" strokeLinecap="round" />
                {/* bottom-left */}
                <path d="M 40 303 L 40 321 L 58 321" stroke="rgba(99,210,255,0.9)" strokeWidth="2.5" fill="none" strokeLinecap="round" />
                {/* bottom-right */}
                <path d="M 260 303 L 260 321 L 242 321" stroke="rgba(99,210,255,0.9)" strokeWidth="2.5" fill="none" strokeLinecap="round" />
            </svg>

            {/* Label */}
            <div style={{
                marginTop: 10,
                background: 'rgba(0,0,0,0.55)',
                backdropFilter: 'blur(8px)',
                color: 'rgba(99,210,255,0.95)',
                fontSize: 12,
                fontWeight: 500,
                padding: '6px 18px',
                borderRadius: 999,
                border: '1px solid rgba(99,210,255,0.3)',
                letterSpacing: '0.5px',
                textTransform: 'uppercase',
            }}>
                Position your face here
            </div>
        </motion.div>
    )
}


// ── Main component ────────────────────────────────────────────────────────────

export default function CameraFeed({
    isActive,
    apiBase,
    userWarnings,
    scanState,
    phase
}) {
    const [cacheBuster, setCacheBuster] = useState(() => Date.now())

    useEffect(() => {
        if (isActive) {
            // eslint-disable-next-line react-hooks/set-state-in-effect
            setCacheBuster(Date.now())
        }
    }, [isActive])

    // ── Warning priority logic ────────────────────────────────────────────────
    // "NO FACE DETECTED" is only shown in IDLE / INITIALIZING (pre-scan).
    // Once FACE_AND_VITALS starts, the backend is already collecting — the
    // banner there is misleading because the debounce may briefly fire on
    // the very first frame even while the face is clearly in view.
    let warningIcon = null
    let warningText = null

    if (userWarnings) {
        if (userWarnings.distance_warning === 'no_face') {
            // Camera is physically blocked or user walked away entirely
            warningIcon = <PersonRoundedIcon />
            warningText = 'NO FACE DETECTED — Position your face in front of the camera'
        } else if (userWarnings.distance_warning === 'too_close') {
            warningIcon = <WarningRoundedIcon />
            warningText = 'TOO CLOSE — Please move back from the camera'
        } else if (userWarnings.distance_warning === 'too_far') {
            warningIcon = <WarningRoundedIcon />
            warningText = 'TOO FAR — Please move closer to the camera'
        } else if (!userWarnings.face_detected && (scanState === 'idle' || phase === 'INITIALIZING')) {
            // Debounce fallback: face consistently absent for 3+ inference frames
            warningIcon = <PersonRoundedIcon />
            warningText = 'NO FACE DETECTED — Position yourself in frame'
        } else if (!userWarnings.pose_detected && phase === 'BODY_ANALYSIS') {
            warningIcon = <PersonRoundedIcon />
            warningText = 'BODY NOT DETECTED — Step back until fully visible'
        } else if (userWarnings.quality_ok === false) {
            warningIcon = <LightbulbRoundedIcon />
            warningText = `POOR SIGNAL QUALITY — Improve lighting or hold still`
        }
    }

    // Show AI spinner overlay during PROCESSING phase
    const showLoaderOverlay = phase === 'PROCESSING'

    // Show the head guide when camera is on and the user needs positioning assistance.
    // Includes FACE_AND_VITALS so users can still adjust during the active face scan.
    const guidePhases = ['IDLE', 'INITIALIZING', 'FACE_AND_VITALS']
    const showHeadGuide = isActive && guidePhases.includes(phase)

    return (
        <div style={{
            flex: 1,
            minHeight: '350px',
            position: 'relative',
            background: '#000',
            borderRadius: '28px',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            aspectRatio: '4/3',
        }}>
            {/* ── LIVE Badge ── */}
            <AnimatePresence>
                {isActive && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        style={{
                            position: 'absolute', top: 24, left: 24, zIndex: 30,
                            background: 'var(--md-error)', color: 'var(--md-on-error)',
                            fontSize: 12, fontWeight: 500, padding: '8px 16px',
                            borderRadius: 999, display: 'flex', alignItems: 'center', gap: 8,
                            letterSpacing: 0.5, boxShadow: '0 2px 6px rgba(0,0,0,0.2)'
                        }}
                    >
                        <motion.div
                            animate={{ opacity: [1, 0.4, 1] }}
                            transition={{ repeat: Infinity, duration: 1.5, ease: 'easeInOut' }}
                            style={{ width: 8, height: 8, background: 'currentColor', borderRadius: '50%' }}
                        />
                        LIVE
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── MJPEG Stream ── */}
            <img
                src={isActive ? `${apiBase}/api/v1/hardware/video-feed?t=${cacheBuster}` : ''}
                alt="Live Camera Feed"
                style={{
                    position: 'absolute', width: '100%', height: '100%',
                    objectFit: 'cover', transform: 'scaleX(-1)',
                    display: isActive ? 'block' : 'none',
                }}
            />

            {/* ── Static Head-Position Guide (pre-scan only) ── */}
            <AnimatePresence>
                {showHeadGuide && <HeadGuide />}
            </AnimatePresence>

            {/* ── Placeholder (When Inactive) ── */}
            <AnimatePresence>
                {!isActive && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{
                            position: 'absolute', zIndex: 20, textAlign: 'center',
                            color: 'var(--md-surface-container-low)', pointerEvents: 'none',
                            background: 'rgba(20,20,20,0.88)', padding: 32,
                            borderRadius: 28,
                        }}
                    >
                        <PhotoCameraRoundedIcon style={{ fontSize: 48, marginBottom: 16, opacity: 0.8 }} />
                        <h2 style={{ fontFamily: "'Google Sans', sans-serif", fontWeight: 400, fontSize: 24, margin: 0 }}>
                            Camera Ready
                        </h2>
                        <p style={{ marginTop: 8, opacity: 0.8, fontSize: 14 }}>
                            Click &apos;Show Camera&apos; or &apos;Start Scan&apos; to begin.
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── AI Processing Overlay ── */}
            <AnimatePresence>
                {showLoaderOverlay && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{
                            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                            zIndex: 50, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(16px)',
                            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                            color: '#fff'
                        }}
                    >
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                            style={{
                                width: 56, height: 56, borderRadius: '50%',
                                border: '6px solid rgba(255,255,255,0.2)',
                                borderTopColor: 'var(--md-primary)',
                                marginBottom: 24
                            }}
                        />
                        <h2 style={{ fontFamily: "'Google Sans', sans-serif", fontSize: 24, margin: '0 0 8px 0', fontWeight: 500 }}>
                            AI Analysis In Progress
                        </h2>
                        <p style={{ opacity: 0.9, fontSize: 14, margin: 0 }}>
                            Generating comprehensive health report...
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Dynamic User Warnings (Distance / Face / Lighting) ── */}
            <AnimatePresence>
                {warningText && !showLoaderOverlay && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        style={{
                            position: 'absolute', bottom: 32, zIndex: 40,
                            background: 'var(--md-error-container)',
                            color: 'var(--md-on-error-container)',
                            padding: '12px 24px', borderRadius: 999,
                            display: 'flex', alignItems: 'center', gap: 12,
                            fontSize: 14, fontWeight: 500,
                            boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                        }}
                    >
                        {warningIcon}
                        {warningText}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

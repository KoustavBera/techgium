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
import SilhouetteOverlay from './SilhouetteOverlay'

// (HeadGuide and inline body SVG have been replaced by <SilhouetteOverlay />)


// ── Main component ────────────────────────────────────────────────────────────

const ACTIVITY_STYLES = {
    still_standing: {
        icon: "🧍",
        title: "Phase 1: Stand Still",
        detail: "Arms relaxed. Look straight ahead. Don't move.",
        color: "#10B981",  // green
    },
    walk_forward_back: {
        icon: "🚶",
        title: "Phase 2: Walk Forward & Back",
        detail: "Walk 3 steps forward, then 3 steps back. Repeat.",
        color: "#3B82F6",  // blue
    }
}

export default function CameraFeed({
    isActive,
    apiBase,
    userWarnings,
    scanState,
    phase,
    activity,
    uiHint,
    phaseIcon,
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

    // ── Silhouette overlay mode ───────────────────────────────────────────────
    const facePhases = ['IDLE', 'INITIALIZING', 'FACE_ADJUST', 'FACE_ADJUST_EXTENDED', 'FACE_AND_VITALS']
    const bodyAdjustPhases = ['BODY_ADJUST', 'BODY_ADJUST_EXTENDED']

    let overlayMode = null
    let isAligned = false

    if (isActive) {
        if (facePhases.includes(phase)) {
            overlayMode = 'face'
            isAligned = !!(userWarnings?.face_detected)
        } else if (bodyAdjustPhases.includes(phase)) {
            overlayMode = 'still'
            isAligned = !!(userWarnings?.pose_detected)
        } else if (phase === 'BODY_ANALYSIS') {
            overlayMode = activity === 'walk_forward_back' ? 'walk' : 'still'
            isAligned = !!(userWarnings?.pose_detected)
        }
    }

    // Show AI spinner overlay during PROCESSING phase
    const showLoaderOverlay = phase === 'PROCESSING'

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

            {/* ── Silhouette Overlay (face, still, walk) ── */}
            <SilhouetteOverlay mode={overlayMode} isAligned={isAligned} />

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

            {/* ── Activity Instruction Banner ── */}
            <AnimatePresence>
                {isActive && phase === 'BODY_ANALYSIS' && activity && ACTIVITY_STYLES[activity] && (
                    <motion.div
                        key={activity}
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 50 }}
                        style={{
                            position: 'absolute', bottom: 40, zIndex: 45,
                            background: 'rgba(20, 20, 20, 0.95)',
                            border: `2px solid ${ACTIVITY_STYLES[activity].color}`,
                            padding: '16px 32px', borderRadius: 24,
                            display: 'flex', alignItems: 'center', gap: 24,
                            boxShadow: `0 8px 32px ${ACTIVITY_STYLES[activity].color}33`,
                            backdropFilter: 'blur(12px)'
                        }}
                    >
                        <span style={{ fontSize: 48, lineHeight: 1 }}>{ACTIVITY_STYLES[activity].icon}</span>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                            <span style={{ 
                                color: ACTIVITY_STYLES[activity].color, 
                                fontSize: 24, fontWeight: 700, fontFamily: "'Google Sans', sans-serif" 
                            }}>
                                {ACTIVITY_STYLES[activity].title}
                            </span>
                            <span style={{ color: '#fff', fontSize: 16, opacity: 0.9 }}>
                                {ACTIVITY_STYLES[activity].detail}
                            </span>
                        </div>
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
                            position: 'absolute', bottom: (phase === 'BODY_ANALYSIS' && activity) ? 160 : 32, zIndex: 40,
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

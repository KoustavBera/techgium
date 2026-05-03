/**
 * SilhouetteOverlay.jsx
 * A dynamic, activity-aware body silhouette overlay for the camera feed.
 *
 * Modes:
 *   face         → Head + shoulder oval (face alignment phases)
 *   still        → Full-body standing silhouette (still_standing activity)
 *   walk         → Full-body silhouette with animated walking legs (walk_forward_back)
 *
 * When isAligned=true the outline turns green + goes semi-transparent to signal
 * good positioning without occluding the live feed.
 */
import { motion, AnimatePresence } from 'framer-motion'

// ── Shared SVG helpers ────────────────────────────────────────────────────────

const STROKE_ALIGNED   = 'rgba(16, 185, 129, 0.85)'  // green
const STROKE_UNALIGNED = 'rgba(99, 210, 255, 0.85)'  // cyan
const STROKE_DIM       = 'rgba(16, 185, 129, 0.35)'  // dim green when aligned

function GlowFilter({ id }) {
    return (
        <defs>
            <filter id={id} x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
        </defs>
    )
}

// ── Face Mode ─────────────────────────────────────────────────────────────────

function FaceSilhouette({ isAligned }) {
    const stroke = isAligned ? STROKE_ALIGNED : STROKE_UNALIGNED

    return (
        <motion.div
            key="face-guide"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: isAligned ? 0.45 : 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.4 }}
            style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                zIndex: 25, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', pointerEvents: 'none',
            }}
        >
            <svg viewBox="0 0 300 420" style={{ width: '65%', maxWidth: 420 }} xmlns="http://www.w3.org/2000/svg">
                <GlowFilter id="glow-face" />
                {/* Shoulder silhouette */}
                <path d="M 0 420 Q 0 375 70 360 Q 150 340 230 360 Q 300 375 300 420 Z"
                    fill="rgba(255,255,255,0.05)" stroke={stroke} strokeWidth="2" strokeDasharray="8 5" />
                {/* Neck */}
                <rect x="127" y="337" width="46" height="38" rx="8"
                    fill="rgba(255,255,255,0.05)" stroke={stroke} strokeWidth="1.5" />
                {/* Head oval */}
                <ellipse cx="150" cy="210" rx="118" ry="138"
                    fill="rgba(255,255,255,0.03)" stroke={stroke} strokeWidth="2.5"
                    strokeDasharray="14 7" filter="url(#glow-face)">
                    <animate attributeName="stroke-dashoffset" from="0" to="-42" dur="2.4s" repeatCount="indefinite" />
                </ellipse>
                {/* Corner brackets */}
                <path d="M 40 125 L 40 107 L 58 107"  stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 260 125 L 260 107 L 242 107" stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 40 303 L 40 321 L 58 321"  stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 260 303 L 260 321 L 242 321" stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
            </svg>
            {!isAligned && (
                <div style={{
                    marginTop: 10, background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)',
                    color: 'rgba(99,210,255,0.95)', fontSize: 12, fontWeight: 500,
                    padding: '6px 18px', borderRadius: 999, border: '1px solid rgba(99,210,255,0.3)',
                    letterSpacing: '0.5px', textTransform: 'uppercase',
                }}>
                    Position your face here
                </div>
            )}
        </motion.div>
    )
}

// ── Still-Standing Mode ───────────────────────────────────────────────────────

function StillSilhouette({ isAligned }) {
    const stroke      = isAligned ? STROKE_ALIGNED   : STROKE_UNALIGNED
    const strokeDim   = isAligned ? STROKE_DIM       : 'rgba(99,210,255,0.55)'
    const strokeMid   = isAligned ? STROKE_DIM       : 'rgba(99,210,255,0.70)'

    return (
        <motion.div
            key="still-guide"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: isAligned ? 0.4 : 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.4 }}
            style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                zIndex: 25, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', pointerEvents: 'none',
            }}
        >
            <svg viewBox="0 0 200 500" style={{ width: '40%', maxWidth: 300 }} xmlns="http://www.w3.org/2000/svg">
                <GlowFilter id="glow-still" />
                {/* Head */}
                <ellipse cx="100" cy="42" rx="34" ry="40"
                    fill="rgba(255,255,255,0.04)" stroke={stroke} strokeWidth="2"
                    strokeDasharray="10 5" filter="url(#glow-still)">
                    <animate attributeName="stroke-dashoffset" from="0" to="-30" dur="2.4s" repeatCount="indefinite" />
                </ellipse>
                {/* Torso */}
                <rect x="62" y="96" width="76" height="160" rx="14"
                    fill="rgba(255,255,255,0.04)" stroke={strokeMid} strokeWidth="1.8" strokeDasharray="10 5" />
                {/* Arms — straight down (at rest) */}
                <rect x="22" y="100" width="34" height="120" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                <rect x="144" y="100" width="34" height="120" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                {/* Legs — parallel, standing */}
                <rect x="64" y="262" width="34" height="180" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                <rect x="102" y="262" width="34" height="180" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                {/* Corner brackets */}
                <path d="M 40 20 L 40 6 L 56 6"     stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 160 20 L 160 6 L 144 6"   stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 40 480 L 40 494 L 56 494" stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 160 480 L 160 494 L 144 494" stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
            </svg>
            {!isAligned && (
                <div style={{
                    marginTop: 12, background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)',
                    color: 'rgba(99,210,255,0.95)', fontSize: 12, fontWeight: 500,
                    padding: '6px 18px', borderRadius: 999, border: '1px solid rgba(99,210,255,0.3)',
                    letterSpacing: '0.5px', textTransform: 'uppercase',
                }}>
                    Step back — full body in frame
                </div>
            )}
        </motion.div>
    )
}

// ── Walking Mode ──────────────────────────────────────────────────────────────
// Uses CSS animation to swing the legs and arms alternately.

const walkCSS = `
@keyframes swingL { 0%,100%{transform-box:fill-box;transform-origin:50% 0%;transform:rotate(-20deg)} 50%{transform:rotate(20deg)} }
@keyframes swingR { 0%,100%{transform-box:fill-box;transform-origin:50% 0%;transform:rotate(20deg)}  50%{transform:rotate(-20deg)} }
.arm-l  { animation: swingL 0.8s ease-in-out infinite }
.arm-r  { animation: swingR 0.8s ease-in-out infinite }
.leg-l  { animation: swingR 0.8s ease-in-out infinite }
.leg-r  { animation: swingL 0.8s ease-in-out infinite }
`

function WalkSilhouette({ isAligned }) {
    const stroke    = isAligned ? STROKE_ALIGNED : STROKE_UNALIGNED
    const strokeDim = isAligned ? STROKE_DIM     : 'rgba(59, 130, 246, 0.6)'

    return (
        <motion.div
            key="walk-guide"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: isAligned ? 0.4 : 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.4 }}
            style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                zIndex: 25, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', pointerEvents: 'none',
            }}
        >
            <style>{walkCSS}</style>
            <svg viewBox="0 0 200 500" style={{ width: '40%', maxWidth: 300 }} xmlns="http://www.w3.org/2000/svg">
                <GlowFilter id="glow-walk" />
                {/* Head */}
                <ellipse cx="100" cy="42" rx="34" ry="40"
                    fill="rgba(255,255,255,0.04)" stroke={stroke} strokeWidth="2"
                    strokeDasharray="10 5" filter="url(#glow-walk)">
                    <animate attributeName="stroke-dashoffset" from="0" to="-30" dur="2.4s" repeatCount="indefinite" />
                </ellipse>
                {/* Torso */}
                <rect x="62" y="96" width="76" height="160" rx="14"
                    fill="rgba(255,255,255,0.04)" stroke="rgba(59,130,246,0.70)" strokeWidth="1.8" strokeDasharray="10 5" />
                {/* Arms — animated */}
                <rect className="arm-l" x="22" y="100" width="34" height="120" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                <rect className="arm-r" x="144" y="100" width="34" height="120" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                {/* Legs — animated */}
                <rect className="leg-l" x="64" y="262" width="34" height="180" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                <rect className="leg-r" x="102" y="262" width="34" height="180" rx="14"
                    fill="rgba(255,255,255,0.03)" stroke={strokeDim} strokeWidth="1.5" strokeDasharray="8 5" />
                {/* Corner brackets */}
                <path d="M 40 20 L 40 6 L 56 6"        stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 160 20 L 160 6 L 144 6"      stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 40 480 L 40 494 L 56 494"    stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
                <path d="M 160 480 L 160 494 L 144 494" stroke={stroke} strokeWidth="2.5" fill="none" strokeLinecap="round" />
            </svg>
            {!isAligned && (
                <div style={{
                    marginTop: 12, background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)',
                    color: 'rgba(59,130,246,0.95)', fontSize: 12, fontWeight: 500,
                    padding: '6px 18px', borderRadius: 999, border: '1px solid rgba(59,130,246,0.3)',
                    letterSpacing: '0.5px', textTransform: 'uppercase',
                }}>
                    Walk forward & back
                </div>
            )}
        </motion.div>
    )
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * @param {string}  mode        'face' | 'still' | 'walk'
 * @param {boolean} isAligned   true when patient is properly positioned
 */
export default function SilhouetteOverlay({ mode, isAligned }) {
    return (
        <AnimatePresence mode="wait">
            {mode === 'face'  && <FaceSilhouette  key="face"  isAligned={isAligned} />}
            {mode === 'still' && <StillSilhouette key="still" isAligned={isAligned} />}
            {mode === 'walk'  && <WalkSilhouette  key="walk"  isAligned={isAligned} />}
        </AnimatePresence>
    )
}

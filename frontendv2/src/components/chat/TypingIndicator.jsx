/**
 * TypingIndicator.jsx
 * Animated 3-dot "thinking" indicator with dynamic agent state text.
 * Dots use CSS keyframe animation (cheaper than Framer Motion JS loop).
 */
import { motion, AnimatePresence } from 'framer-motion'

const dotStyle = (delayClass) => ({
    display: 'block',
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    background: 'var(--md-primary)',
})

export default function TypingIndicator({ visible, label }) {
    return (
        <AnimatePresence>
            {visible && (
                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 6 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 22 }}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        padding: '10px 16px',
                        alignSelf: 'flex-start',
                        maxWidth: '320px',
                    }}
                >
                    {/* Dots container — uses CSS @keyframes for off-thread animation */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px',
                        padding: '10px 14px',
                        background: 'var(--md-surface-container)',
                        borderRadius: '8px 20px 20px 20px',
                        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                    }}>
                        <span className="animate-typing-dot" style={dotStyle()} />
                        <span className="animate-typing-dot" style={dotStyle()} />
                        <span className="animate-typing-dot" style={dotStyle()} />
                    </div>
                    {/* Label */}
                    {label && (
                        <span style={{
                            fontFamily: "'Google Sans', sans-serif",
                            fontSize: '12px',
                            color: 'var(--md-on-surface-variant)',
                            fontStyle: 'italic',
                        }}>
                            {label}
                        </span>
                    )}
                </motion.div>
            )}
        </AnimatePresence>
    )
}

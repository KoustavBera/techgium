/**
 * TypingIndicator.jsx
 * Animated 3-dot "thinking" indicator with dynamic agent state text.
 * Dots use CSS keyframe animation (cheaper than Framer Motion JS loop).
 */
import { motion, AnimatePresence } from 'framer-motion'


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
                    {/* Thick Logo-style gradient circle */}
                    <motion.div
                        animate={{ 
                            opacity: [0.7, 1, 0.7],
                            scale: [0.95, 1.05, 0.95]
                        }}
                        transition={{ 
                            duration: 1.5, 
                            ease: 'easeInOut', 
                            repeat: Infinity 
                        }}
                        style={{
                            width: '24px',
                            height: '24px',
                            borderRadius: '50%',
                            background: 'linear-gradient(45deg, #FE3290 0%, #3E88FD 100%)',
                            boxShadow: '0 2px 6px rgba(0,0,0,0.1)'
                        }}
                    />
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

/**
 * TypingIndicator.jsx
 * Animated 3-dot "thinking" indicator with dynamic agent state text.
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
                    {/* Dots container */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px',
                        padding: '10px 14px',
                        background: 'var(--md-surface-container)',
                        borderRadius: '8px 20px 20px 20px',
                        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                    }}>
                        {[0, 1, 2].map(i => (
                            <motion.span
                                key={i}
                                animate={{ y: [0, -6, 0] }}
                                transition={{
                                    duration: 0.8,
                                    repeat: Infinity,
                                    delay: i * 0.15,
                                    ease: 'easeInOut',
                                }}
                                style={{
                                    display: 'block',
                                    width: '7px',
                                    height: '7px',
                                    borderRadius: '50%',
                                    background: 'var(--md-primary)',
                                }}
                            />
                        ))}
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

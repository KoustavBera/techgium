/**
 * LanguageOverlay.jsx
 * Full-screen frosted glass language picker — shown once per session.
 */
import { motion, AnimatePresence } from 'framer-motion'

const LANGUAGES = [
    { code: 'en-IN', native: 'English', english: 'English' },
    { code: 'hi-IN', native: 'हिंदी', english: 'Hindi' },
    { code: 'bn-IN', native: 'বাংলা', english: 'Bengali' },
    { code: 'kn-IN', native: 'ಕನ್ನಡ', english: 'Kannada' },
]

const cardSpring = { type: 'spring', stiffness: 320, damping: 24 }
const overlaySpring = { type: 'spring', stiffness: 280, damping: 28 }

export default function LanguageOverlay({ visible, onSelect }) {
    return (
        <AnimatePresence>
            {visible && (
                <motion.div
                    key="lang-overlay"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0, scale: 1.04 }}
                    transition={{ duration: 0.22 }}
                    style={{
                        position: 'absolute',
                        inset: 0,
                        zIndex: 50,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderRadius: 'inherit',
                        overflow: 'hidden',
                    }}
                    className="glass-overlay"
                >
                    {/* Inner card */}
                    <motion.div
                        initial={{ scale: 0.92, opacity: 0, y: 20 }}
                        animate={{ scale: 1, opacity: 1, y: 0 }}
                        exit={{ scale: 0.96, opacity: 0, y: 10 }}
                        transition={overlaySpring}
                        style={{ textAlign: 'center', padding: '8px 16px 24px' }}
                    >
                        {/* Title */}
                        <h2 style={{
                            fontFamily: "'Google Sans Display', 'Google Sans', sans-serif",
                            fontSize: '26px', fontWeight: 500,
                            color: 'var(--md-on-surface)',
                            marginBottom: '6px',
                        }}>
                            Welcome to Chiranjeevi
                        </h2>
                        <p style={{
                            fontFamily: "'Google Sans', sans-serif",
                            fontSize: '14px',
                            color: 'var(--md-on-surface-variant)',
                            marginBottom: '32px',
                        }}>
                            Please select your preferred language<br />
                            <span style={{ fontSize: '13px', opacity: 0.75 }}>
                                कृपया अपनी पसंदीदा भाषा चुनें
                            </span>
                        </p>

                        {/* Language grid */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(2, 1fr)',
                            gap: '14px',
                            maxWidth: '440px',
                            width: '90vw',
                        }}>
                            {LANGUAGES.map((lang, i) => (
                                <motion.button
                                    key={lang.code}
                                    initial={{ opacity: 0, y: 16 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ ...cardSpring, delay: 0.08 + i * 0.06 }}
                                    whileHover={{ y: -4, scale: 1.03, boxShadow: '0 8px 24px rgba(11,87,208,0.14)' }}
                                    whileTap={{ scale: 0.97 }}
                                    onClick={() => onSelect(lang.code)}
                                    style={{
                                        background: 'var(--md-surface-container-lowest)',
                                        border: '1.5px solid var(--md-outline-variant)',
                                        borderRadius: '18px',
                                        padding: '20px 16px',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        gap: '4px',
                                        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                                    }}
                                >
                                    <span style={{
                                        fontFamily: "'Google Sans', sans-serif",
                                        fontSize: '22px', fontWeight: 500,
                                        color: 'var(--md-on-surface)',
                                    }}>
                                        {lang.native}
                                    </span>
                                    <span style={{
                                        fontFamily: "'Google Sans', sans-serif",
                                        fontSize: '11px', fontWeight: 500,
                                        color: 'var(--md-on-surface-variant)',
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.6px',
                                    }}>
                                        {lang.english}
                                    </span>
                                </motion.button>
                            ))}
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    )
}

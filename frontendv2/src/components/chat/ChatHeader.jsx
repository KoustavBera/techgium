/**
 * ChatHeader.jsx
 * Title bar with connection status badge + active language pill.
 */
import { motion } from 'framer-motion'
import SmartToyRoundedIcon from '@mui/icons-material/SmartToyRounded'
import TranslateRoundedIcon from '@mui/icons-material/TranslateRounded'

const LANG_NAMES = {
    'en-IN': 'English',
    'hi-IN': 'हिंदी',
    'bn-IN': 'বাংলা',
    'kn-IN': 'ಕನ್ನಡ',
}

const STATUS_CONFIG = {
    connected: { bg: 'var(--md-success-container)', color: 'var(--md-on-success-container)', label: '● Connected' },
    disconnected: { bg: 'var(--md-error-container)', color: 'var(--md-on-error-container)', label: '● Disconnected' },
    checking: { bg: 'var(--md-warning-container)', color: 'var(--md-on-warning-container)', label: '○ Checking…' },
}

export default function ChatHeader({ connectionStatus, language, onChangeLang }) {
    const status = STATUS_CONFIG[connectionStatus] || STATUS_CONFIG.checking
    const langName = LANG_NAMES[language] || 'English'

    return (
        <div style={{
            padding: '20px 24px',
            background: 'white',
            borderRadius: '24px 24px 0 0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexShrink: 0,
            borderBottom: '1px solid var(--md-surface-container-high)',
        }}>
            {/* Left: title + subtitle */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{
                    width: 40, height: 40,
                    borderRadius: '14px',
                    background: 'var(--md-tertiary-container)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                    <SmartToyRoundedIcon style={{ fontSize: 22, color: 'var(--md-on-tertiary-container)' }} />
                </div>
                <div>
                    <div style={{
                        fontFamily: "'Google Sans Display', 'Google Sans', sans-serif",
                        fontSize: '18px', fontWeight: 500,
                        color: 'var(--md-on-surface)',
                        letterSpacing: '-0.2px',
                    }}>
                        CHIRANJEEVAI
                    </div>
                    <div style={{
                        fontFamily: "'Google Sans', sans-serif",
                        fontSize: '12px',
                        color: 'var(--md-on-surface-variant)',
                    }}>
                        AI Medical Assistant
                    </div>
                </div>
            </div>

            {/* Right: status + language pill */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {/* Connection badge */}
                <motion.div
                    key={connectionStatus}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    style={{
                        padding: '5px 12px',
                        borderRadius: '999px',
                        background: status.bg,
                        color: status.color,
                        fontSize: '11px', fontWeight: 600,
                        fontFamily: "'Google Sans', sans-serif",
                        letterSpacing: '0.2px',
                    }}
                >
                    {status.label}
                </motion.div>

                {/* Language pill (clickable) */}
                <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={onChangeLang}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        padding: '6px 14px',
                        borderRadius: '999px',
                        background: 'var(--md-primary-container)',
                        color: 'var(--md-on-primary-container)',
                        border: 'none', cursor: 'pointer',
                        fontSize: '12px', fontWeight: 600,
                        fontFamily: "'Google Sans', sans-serif",
                    }}
                >
                    <TranslateRoundedIcon style={{ fontSize: 14 }} />
                    {langName}
                </motion.button>
            </div>
        </div>
    )
}

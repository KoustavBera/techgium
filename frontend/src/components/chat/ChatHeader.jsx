/**
 * ChatHeader.jsx
 * Title bar with connection status badge + active language pill.
 */
import { motion } from 'framer-motion'
import SmartToyRoundedIcon from '@mui/icons-material/SmartToyRounded'
import TranslateRoundedIcon from '@mui/icons-material/TranslateRounded'
import { LANGUAGES } from './LanguageOverlay'

// Build lookup from the single source-of-truth list in LanguageOverlay
const LANG_META = Object.fromEntries(LANGUAGES.map(l => [l.code, l]))

const STATUS_CONFIG = {
    connected: { bg: 'var(--md-success-container)', color: 'var(--md-on-success-container)', label: '● Connected' },
    disconnected: { bg: 'var(--md-error-container)', color: 'var(--md-on-error-container)', label: '● Disconnected' },
    checking: { bg: 'var(--md-warning-container)', color: 'var(--md-on-warning-container)', label: '○ Checking…' },
}

export default function ChatHeader({ connectionStatus, language, onChangeLang }) {
    const status   = STATUS_CONFIG[connectionStatus] || STATUS_CONFIG.checking
    const langMeta = LANG_META[language] || LANG_META['en-IN']
    const langName = langMeta?.native || 'English'
    const langFlag = langMeta?.flag   || '🌐'
    const langColor = langMeta?.color || 'var(--md-primary)'

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
                    title="Change language"
                    aria-label={`Current language: ${langMeta?.english}. Click to change.`}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        padding: '5px 12px 5px 10px',
                        borderRadius: '999px',
                        background: `${langColor}15`,
                        color: langColor,
                        border: `1.5px solid ${langColor}35`,
                        cursor: 'pointer',
                        fontSize: '12px', fontWeight: 600,
                        fontFamily: "'Google Sans', sans-serif",
                    }}
                >
                    <span style={{ fontSize: 14, lineHeight: 1 }} aria-hidden>{langFlag}</span>
                    <span>{langName}</span>
                    <TranslateRoundedIcon style={{ fontSize: 12, opacity: 0.7 }} />
                </motion.button>
            </div>
        </div>
    )
}

/**
 * CitationPill.jsx
 * Compact "N sources" pill shown below assistant messages.
 */
import { motion } from 'framer-motion'
import SourceRoundedIcon from '@mui/icons-material/SourceRounded'

export default function CitationPill({ citations, onClick }) {
    if (!citations || citations.length === 0) return null

    return (
        <motion.button
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 340, damping: 24 }}
            onClick={onClick}
            style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '5px 12px',
                borderRadius: '999px',
                background: 'var(--md-secondary-container)',
                color: 'var(--md-on-secondary-container)',
                border: 'none',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 600,
                fontFamily: "'Google Sans', sans-serif",
                marginTop: '8px',
                letterSpacing: '0.1px',
            }}
        >
            <SourceRoundedIcon style={{ fontSize: 14 }} />
            {citations.length} source{citations.length !== 1 ? 's' : ''}
        </motion.button>
    )
}

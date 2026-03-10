/**
 * CitationSidebar.jsx
 * Spring-animated slide-out panel showing message sources.
 */
import { motion, AnimatePresence } from 'framer-motion'
import CloseRoundedIcon from '@mui/icons-material/CloseRounded'
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded'
import ArticleRoundedIcon from '@mui/icons-material/ArticleRounded'
import PublicRoundedIcon from '@mui/icons-material/PublicRounded'
import ScienceRoundedIcon from '@mui/icons-material/ScienceRounded'

const sidebarSpring = { type: 'spring', stiffness: 300, damping: 28 }

function getSourceIcon(source) {
    const s = (source || '').toLowerCase()
    if (s.includes('pubmed') || s.includes('ncbi') || s.includes('medline')) return ScienceRoundedIcon
    if (s.includes('web') || s.includes('search')) return PublicRoundedIcon
    return ArticleRoundedIcon
}

export default function CitationSidebar({ citations, onClose }) {
    return (
        <AnimatePresence>
            {citations && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        key="citation-backdrop"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        onClick={onClose}
                        style={{
                            position: 'absolute', inset: 0,
                            background: 'rgba(0,0,0,0.12)',
                            zIndex: 30,
                        }}
                    />

                    {/* Panel */}
                    <motion.div
                        key="citation-panel"
                        initial={{ x: '100%', opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: '100%', opacity: 0 }}
                        transition={sidebarSpring}
                        style={{
                            position: 'absolute',
                            right: 0, top: 0, bottom: 0,
                            width: 'min(360px, 90%)',
                            background: 'var(--md-surface-container-lowest)',
                            boxShadow: '-4px 0 24px rgba(0,0,0,0.10)',
                            zIndex: 40,
                            display: 'flex', flexDirection: 'column',
                            borderRadius: '0 24px 24px 0',
                            overflow: 'hidden',
                        }}
                    >
                        {/* Header */}
                        <div style={{
                            padding: '20px 20px 16px',
                            borderBottom: '1px solid var(--md-surface-container-high)',
                            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                            flexShrink: 0,
                        }}>
                            <span style={{
                                fontFamily: "'Google Sans Display', 'Google Sans', sans-serif",
                                fontSize: '17px', fontWeight: 500,
                                color: 'var(--md-on-surface)',
                            }}>
                                Sources ({citations.length})
                            </span>
                            <motion.button
                                whileHover={{ scale: 1.1, rotate: 90 }}
                                whileTap={{ scale: 0.95 }}
                                transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                                onClick={onClose}
                                style={{
                                    width: 36, height: 36, borderRadius: '50%',
                                    border: 'none', cursor: 'pointer',
                                    background: 'var(--md-surface-container)',
                                    color: 'var(--md-on-surface-variant)',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }}
                            >
                                <CloseRoundedIcon style={{ fontSize: 18 }} />
                            </motion.button>
                        </div>

                        {/* Citations list */}
                        <div style={{
                            flex: 1, overflowY: 'auto', padding: '16px',
                            display: 'flex', flexDirection: 'column', gap: '12px',
                        }}>
                            {citations.map((cite, i) => {
                                const Icon = getSourceIcon(cite.source || '')
                                const title = cite.title || cite.url || `Source ${i + 1}`
                                const url = cite.url || cite.link || null

                                return (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, x: 16 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ ...sidebarSpring, delay: i * 0.05 }}
                                        style={{
                                            padding: '14px',
                                            background: 'var(--md-surface-container-low)',
                                            borderRadius: '14px',
                                            display: 'flex', flexDirection: 'column', gap: '6px',
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                                            <Icon style={{
                                                fontSize: 16,
                                                color: 'var(--md-primary)',
                                                flexShrink: 0, marginTop: 2,
                                            }} />
                                            <span style={{
                                                fontFamily: "'Google Sans', sans-serif",
                                                fontSize: '13px', fontWeight: 500,
                                                color: 'var(--md-on-surface)',
                                                lineHeight: 1.4,
                                                display: '-webkit-box',
                                                WebkitLineClamp: 3,
                                                WebkitBoxOrient: 'vertical',
                                                overflow: 'hidden',
                                            }}>
                                                {title}
                                            </span>
                                        </div>
                                        {url && (
                                            <a
                                                href={url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                style={{
                                                    display: 'flex', alignItems: 'center', gap: '4px',
                                                    fontSize: '11px',
                                                    color: 'var(--md-primary)',
                                                    textDecoration: 'none',
                                                    paddingLeft: '24px',
                                                }}
                                            >
                                                <OpenInNewRoundedIcon style={{ fontSize: 12 }} />
                                                <span style={{
                                                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                                    maxWidth: '240px',
                                                }}>
                                                    {url.replace(/^https?:\/\//, '')}
                                                </span>
                                            </a>
                                        )}
                                    </motion.div>
                                )
                            })}
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    )
}

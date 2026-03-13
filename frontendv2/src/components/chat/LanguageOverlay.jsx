/**
 * LanguageOverlay.jsx
 * Full-screen frosted glass language picker — shown once per session.
 * Supports all 11 Sarvam AI languages with search, keyboard navigation, and beautiful UI.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export const LANGUAGES = [
    { code: 'en-IN', native: 'English',    english: 'English',   script: 'Latin',      flag: '🇮🇳', color: '#4285F4' },
    { code: 'hi-IN', native: 'हिंदी',      english: 'Hindi',     script: 'Devanagari', flag: '🕉️', color: '#F4B400' },
    { code: 'bn-IN', native: 'বাংলা',      english: 'Bengali',   script: 'Bengali',    flag: '🌿', color: '#34A853' },
    { code: 'ta-IN', native: 'தமிழ்',     english: 'Tamil',     script: 'Tamil',      flag: '🌺', color: '#EA4335' },
    { code: 'te-IN', native: 'తెలుగు',    english: 'Telugu',    script: 'Telugu',     flag: '🌸', color: '#FF6D00' },
    { code: 'mr-IN', native: 'मराठी',      english: 'Marathi',   script: 'Devanagari', flag: '🛕', color: '#9C27B0' },
    { code: 'gu-IN', native: 'ગુજરાતી',   english: 'Gujarati',  script: 'Gujarati',   flag: '🦁', color: '#00897B' },
    { code: 'kn-IN', native: 'ಕನ್ನಡ',     english: 'Kannada',   script: 'Kannada',    flag: '🐘', color: '#D81B60' },
    { code: 'ml-IN', native: 'മലയാളം',    english: 'Malayalam', script: 'Malayalam',  flag: '🌴', color: '#1E88E5' },
    { code: 'pa-IN', native: 'ਪੰਜਾਬੀ',   english: 'Punjabi',   script: 'Gurmukhi',   flag: '🌾', color: '#FF8F00' },
    { code: 'od-IN', native: 'ଓଡ଼ିଆ',      english: 'Odia',      script: 'Odia',       flag: '🐚', color: '#43A047' },
]

const overlaySpring = { type: 'spring', stiffness: 280, damping: 28 }
const cardSpring    = { type: 'spring', stiffness: 350, damping: 26 }

export default function LanguageOverlay({ visible, onSelect }) {
    const [search,    setSearch]    = useState('')
    const [focusIdx,  setFocusIdx]  = useState(-1)
    const searchRef = useRef(null)
    const gridRef   = useRef(null)

    // Auto-focus search box when overlay appears
    useEffect(() => {
        if (visible) {
            setSearch('')
            setFocusIdx(-1)
            setTimeout(() => searchRef.current?.focus(), 120)
        }
    }, [visible])

    const filtered = LANGUAGES.filter(l =>
        !search.trim() ||
        l.english.toLowerCase().includes(search.toLowerCase()) ||
        l.native.toLowerCase().includes(search.toLowerCase())
    )

    // Keyboard navigation
    const handleSearchKey = useCallback((e) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); setFocusIdx(0) }
        else if (e.key === 'Enter' && filtered.length === 1) { onSelect(filtered[0].code) }
    }, [filtered, onSelect])

    const handleCardKey = useCallback((e, lang, idx) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(lang.code) }
        else if (e.key === 'ArrowDown')  { e.preventDefault(); setFocusIdx(Math.min(idx + 1, filtered.length - 1)) }
        else if (e.key === 'ArrowUp')    { e.preventDefault(); idx === 0 ? searchRef.current?.focus() : setFocusIdx(idx - 1) }
        else if (e.key === 'ArrowRight') { e.preventDefault(); setFocusIdx(Math.min(idx + 1, filtered.length - 1)) }
        else if (e.key === 'ArrowLeft')  { e.preventDefault(); setFocusIdx(Math.max(idx - 1, 0)) }
        else if (e.key === 'Escape')     { searchRef.current?.focus(); setFocusIdx(-1) }
    }, [filtered, onSelect])

    return (
        <AnimatePresence>
            {visible && (
                <motion.div
                    key="lang-overlay"
                    role="dialog"
                    aria-modal="true"
                    aria-label="Language selection"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
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
                        background: 'rgba(255,255,255,0.72)',
                        backdropFilter: 'blur(24px) saturate(1.6)',
                        WebkitBackdropFilter: 'blur(24px) saturate(1.6)',
                    }}
                >
                    {/* Decorative gradient orbs */}
                    <div aria-hidden style={{
                        position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden',
                    }}>
                        <div style={{
                            position: 'absolute', width: 320, height: 320, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(66,133,244,0.14) 0%, transparent 70%)',
                            top: '-60px', left: '-80px',
                        }} />
                        <div style={{
                            position: 'absolute', width: 280, height: 280, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(234,67,53,0.10) 0%, transparent 70%)',
                            bottom: '-40px', right: '-60px',
                        }} />
                        <div style={{
                            position: 'absolute', width: 200, height: 200, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(52,168,83,0.10) 0%, transparent 70%)',
                            top: '30%', right: '-40px',
                        }} />
                    </div>

                    {/* Inner card */}
                    <motion.div
                        initial={{ scale: 0.92, opacity: 0, y: 24 }}
                        animate={{ scale: 1, opacity: 1, y: 0 }}
                        exit={{ scale: 0.96, opacity: 0, y: 12 }}
                        transition={overlaySpring}
                        style={{
                            width: '92%',
                            maxWidth: 560,
                            maxHeight: 'calc(100% - 40px)',
                            display: 'flex',
                            flexDirection: 'column',
                            background: 'rgba(255,255,255,0.90)',
                            borderRadius: 24,
                            boxShadow: '0 8px 40px rgba(0,0,0,0.12), 0 1px 4px rgba(0,0,0,0.06)',
                            border: '1px solid rgba(255,255,255,0.8)',
                            overflow: 'hidden',
                        }}
                    >
                        {/* Header */}
                        <div style={{ padding: '28px 28px 0', flexShrink: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                                <span style={{ fontSize: 28 }} aria-hidden>🌐</span>
                                <h2 style={{
                                    fontFamily: "'Google Sans Display', 'Google Sans', sans-serif",
                                    fontSize: 22, fontWeight: 600, margin: 0,
                                    color: 'var(--md-on-surface)',
                                    letterSpacing: '-0.3px',
                                }}>
                                    Choose Your Language
                                </h2>
                            </div>
                            <p style={{
                                fontFamily: "'Google Sans', sans-serif",
                                fontSize: 13, color: 'var(--md-on-surface-variant)',
                                margin: '0 0 18px 0', lineHeight: 1.5,
                            }}>
                                Select the language for your conversation — the AI will respond in your language.
                                <br />
                                <span style={{ opacity: 0.75 }}>अपनी भाषा चुनें • আপনার ভাষা বেছে নিন</span>
                            </p>

                            {/* Search box */}
                            <div style={{ position: 'relative', marginBottom: 16 }}>
                                <span style={{
                                    position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                                    fontSize: 16, pointerEvents: 'none', color: 'var(--md-on-surface-variant)',
                                }} aria-hidden>🔍</span>
                                <input
                                    ref={searchRef}
                                    type="search"
                                    placeholder="Search language…"
                                    value={search}
                                    onChange={e => { setSearch(e.target.value); setFocusIdx(-1) }}
                                    onKeyDown={handleSearchKey}
                                    aria-label="Search languages"
                                    style={{
                                        width: '100%', boxSizing: 'border-box',
                                        padding: '10px 14px 10px 38px',
                                        borderRadius: 999, border: '1.5px solid var(--md-outline-variant)',
                                        background: 'var(--md-surface-container-lowest)',
                                        fontFamily: "'Google Sans', sans-serif",
                                        fontSize: 14, color: 'var(--md-on-surface)',
                                        outline: 'none',
                                        transition: 'border-color 0.18s',
                                        appearance: 'none',
                                    }}
                                    onFocus={e => e.target.style.borderColor = 'var(--md-primary)'}
                                    onBlur={e => e.target.style.borderColor = 'var(--md-outline-variant)'}
                                />
                            </div>

                            {/* Divider */}
                            <div style={{ height: 1, background: 'var(--md-outline-variant)', opacity: 0.5, margin: '0 -28px' }} />
                        </div>

                        {/* Scrollable language grid */}
                        <div
                            ref={gridRef}
                            style={{
                                padding: '16px 28px 24px',
                                overflowY: 'auto',
                                flexGrow: 1,
                                scrollbarWidth: 'thin',
                                scrollbarColor: 'var(--md-outline-variant) transparent',
                            }}
                        >
                            {filtered.length === 0 ? (
                                <div style={{
                                    textAlign: 'center', padding: '32px 0',
                                    color: 'var(--md-on-surface-variant)',
                                    fontFamily: "'Google Sans', sans-serif",
                                    fontSize: 14,
                                }}>
                                    No language found for "{search}"
                                </div>
                            ) : (
                                <div
                                    role="listbox"
                                    aria-label="Available languages"
                                    style={{
                                        display: 'grid',
                                        gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
                                        gap: 10,
                                    }}
                                >
                                    {filtered.map((lang, i) => (
                                        <LanguageCard
                                            key={lang.code}
                                            lang={lang}
                                            index={i}
                                            isFocused={focusIdx === i}
                                            onSelect={onSelect}
                                            onKeyDown={handleCardKey}
                                            cardSpring={cardSpring}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    )
}

function LanguageCard({ lang, index, isFocused, onSelect, onKeyDown, cardSpring }) {
    const ref = useRef(null)

    useEffect(() => {
        if (isFocused) ref.current?.focus()
    }, [isFocused])

    return (
        <motion.button
            ref={ref}
            role="option"
            aria-selected={false}
            aria-label={`${lang.english} — ${lang.native}`}
            tabIndex={isFocused || index === 0 ? 0 : -1}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...cardSpring, delay: Math.min(index * 0.03, 0.18) }}
            whileHover={{ y: -3, scale: 1.03, boxShadow: `0 6px 20px ${lang.color}28` }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSelect(lang.code)}
            onKeyDown={e => onKeyDown(e, lang, index)}
            style={{
                background: 'var(--md-surface-container-lowest)',
                border: `1.5px solid ${lang.color}30`,
                borderRadius: 16,
                padding: '14px 10px',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 5,
                boxShadow: `0 2px 8px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.6)`,
                transition: 'border-color 0.18s',
                position: 'relative',
                overflow: 'hidden',
            }}
        >
            {/* Subtle top accent bar */}
            <div aria-hidden style={{
                position: 'absolute', top: 0, left: 0, right: 0, height: 3,
                borderRadius: '16px 16px 0 0',
                background: `linear-gradient(90deg, ${lang.color}80, ${lang.color}20)`,
            }} />

            {/* Flag/emoji */}
            <span style={{ fontSize: 26, lineHeight: 1, marginTop: 2 }} aria-hidden>
                {lang.flag}
            </span>

            {/* Native script name */}
            <span style={{
                fontFamily: "'Google Sans', Noto Sans, sans-serif",
                fontSize: 15, fontWeight: 600,
                color: 'var(--md-on-surface)',
                lineHeight: 1.2,
                textAlign: 'center',
            }}>
                {lang.native}
            </span>

            {/* English name */}
            <span style={{
                fontFamily: "'Google Sans', sans-serif",
                fontSize: 10.5, fontWeight: 500,
                color: lang.color,
                textTransform: 'uppercase',
                letterSpacing: '0.7px',
            }}>
                {lang.english}
            </span>

            {/* Script label */}
            <span style={{
                fontFamily: "'Google Sans', sans-serif",
                fontSize: 9.5, fontWeight: 400,
                color: 'var(--md-on-surface-variant)',
                opacity: 0.65,
                marginTop: -2,
            }}>
                {lang.script}
            </span>
        </motion.button>
    )
}

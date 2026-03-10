import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import flowerPng from '../assets/flower_png.png'
import BiotechRoundedIcon from '@mui/icons-material/BiotechRounded'
import SmartToyRoundedIcon from '@mui/icons-material/SmartToyRounded'
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded'
import CameraAltRoundedIcon from '@mui/icons-material/CameraAltRounded'
import WifiTetheringRoundedIcon from '@mui/icons-material/WifiTetheringRounded'
import ThermostatRoundedIcon from '@mui/icons-material/ThermostatRounded'
import TranslateRoundedIcon from '@mui/icons-material/TranslateRounded'
import ShieldRoundedIcon from '@mui/icons-material/ShieldRounded'
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded'

/* ── Spring presets ─────────────────────────────────────────── */
const spring = (s = 240, d = 26) => ({ type: 'spring', stiffness: s, damping: d })

/* ── Stat items ─────────────────────────────────────────────── */
const STATS = [
    { value: '3+', label: 'Sensor Modalities', sub: 'Camera · Thermal · mmWave' },
    { value: '7+', label: 'Languages Supported', sub: 'Including Kannada, Hindi & more' },
    { value: '<2s', label: 'Analysis Latency', sub: 'Real-time multimodal inference' },
    { value: '20+', label: 'Biomarkers Tracked', sub: 'HR · SpO₂ · Resp · Temp · HRV …' },
]

/* ── Feature tags ───────────────────────────────────────────── */
const TAGS = [
    { icon: CameraAltRoundedIcon, label: 'RGB Camera' },
    { icon: WifiTetheringRoundedIcon, label: 'mmWave Radar' },
    { icon: ThermostatRoundedIcon, label: 'Thermal Imaging' },
    { icon: TranslateRoundedIcon, label: 'Multilingual AI' },
    { icon: ShieldRoundedIcon, label: 'Privacy-First' },
    { icon: SpeedRoundedIcon, label: 'Real-time' },
]

/* ── Module definitions ─────────────────────────────────────── */
const MODULES = [
    {
        Icon: BiotechRoundedIcon,
        iconBg: 'linear-gradient(135deg, #C2E7FF 0%, #D3E3FD 100%)',
        iconColor: 'var(--md-on-secondary-container)',
        title: 'Health Screening',
        subtitle: 'Non-invasive · Multimodal · Real-time',
        description:
            'Comprehensive vitals and body assessment without any contact. Powered by three sensor modalities working in parallel.',
        features: [
            'Heart rate & SpO₂ via rPPG camera',
            'Respiratory rate & HRV via mmWave radar',
            'Surface temperature via thermal imaging',
            'AI-generated clinical report in seconds',
        ],
        to: '/screening',
        borderRadius: '32px 32px 16px 32px',
        delay: 0.18,
        accentColor: 'rgba(0, 90, 153, 0.22)',
        glowColor: 'rgba(11, 87, 208, 0.15)',
        badgeColor: 'var(--md-secondary-container)',
        badgeText: 'var(--md-on-secondary-container)',
        badge: 'Live Demo',
    },
    {
        Icon: SmartToyRoundedIcon,
        iconBg: 'linear-gradient(135deg, #EADDFF 0%, #D4BBFF 100%)',
        iconColor: 'var(--md-on-tertiary-container)',
        title: 'CHIRANJEEVAI',
        subtitle: 'Evidence-based · Multilingual · Empathetic',
        description:
            'Your AI medical companion — ask health questions, get report summaries, and receive evidence-backed clinical guidance in your language.',
        features: [
            'Interprets your screening report in plain language',
            'Answers evidence-based health questions 24/7',
            'Supports 7+ Indic languages via Sarvam AI',
            'Cites sources with every clinical claim',
        ],
        to: '/chatbot',
        borderRadius: '32px 16px 32px 32px',
        delay: 0.28,
        accentColor: 'rgba(103, 80, 164, 0.22)',
        glowColor: 'rgba(103, 80, 164, 0.15)',
        badgeColor: 'var(--md-tertiary-container)',
        badgeText: 'var(--md-on-tertiary-container)',
        badge: 'AI Assistant',
    },
]

/* ── Sub-components ─────────────────────────────────────────── */

function StatCard({ value, label, sub, index }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...spring(280, 24), delay: 0.32 + index * 0.07 }}
            style={{
                flex: 1,
                minWidth: 0,
                padding: '18px 20px',
                background: 'rgba(255,255,255,0.60)',
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                borderRadius: '20px',
                border: '1px solid rgba(196,199,197,0.4)',
                textAlign: 'center',
            }}
        >
            <div style={{
                fontFamily: "'Google Sans Display','Google Sans',sans-serif",
                fontSize: 'clamp(22px,2.2vw,30px)',
                fontWeight: 700,
                color: 'var(--md-primary)',
                letterSpacing: '-0.5px',
                lineHeight: 1,
                marginBottom: '6px',
            }}>{value}</div>
            <div style={{
                fontFamily: "'Roboto Flex',sans-serif",
                fontSize: '13px',
                fontWeight: 600,
                color: 'var(--md-on-surface)',
                marginBottom: '3px',
            }}>{label}</div>
            <div style={{
                fontFamily: "'Roboto Flex',sans-serif",
                fontSize: '11px',
                color: 'var(--md-on-surface-variant)',
                lineHeight: 1.4,
            }}>{sub}</div>
        </motion.div>
    )
}

function FeatureTag({ Icon, label, index }) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ ...spring(320, 22), delay: 0.55 + index * 0.05 }}
            style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 14px 6px 10px',
                background: 'rgba(255,255,255,0.55)',
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                border: '1px solid rgba(196,199,197,0.45)',
                borderRadius: '999px',
                fontSize: '12px',
                fontWeight: 500,
                color: 'var(--md-on-surface-variant)',
                fontFamily: "'Roboto Flex',sans-serif",
                whiteSpace: 'nowrap',
            }}
        >
            <Icon style={{ fontSize: 14, color: 'var(--md-primary)', opacity: 0.85 }} />
            {label}
        </motion.div>
    )
}

function ModuleCard({ Icon, iconBg, iconColor, title, subtitle, description, features, to, borderRadius, delay, accentColor, glowColor, badgeColor, badgeText, badge }) {
    const [hovered, setHovered] = useState(false)

    return (
        <motion.div
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...spring(280, 22), delay }}
            style={{ height: '100%' }}
        >
            <Link to={to} style={{ textDecoration: 'none', display: 'block', height: '100%' }}>
                <motion.div
                    onHoverStart={() => setHovered(true)}
                    onHoverEnd={() => setHovered(false)}
                    whileHover={{ y: -8, scale: 1.015, boxShadow: `0 24px 60px ${accentColor}, 0 4px 16px rgba(0,0,0,0.06)` }}
                    whileTap={{ scale: 0.98 }}
                    transition={spring(300, 22)}
                    style={{
                        background: 'rgba(255,255,255,0.62)',
                        backdropFilter: 'blur(20px)',
                        WebkitBackdropFilter: 'blur(20px)',
                        borderRadius,
                        padding: '32px',
                        cursor: 'pointer',
                        position: 'relative',
                        overflow: 'hidden',
                        boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
                        border: '1px solid rgba(196,199,197,0.35)',
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '20px',
                        boxSizing: 'border-box',
                    }}
                >
                    {/* Glow blob */}
                    <motion.div
                        animate={{ opacity: hovered ? 0.75 : 0.35, scale: hovered ? 1.15 : 1 }}
                        transition={spring(140, 22)}
                        style={{
                            position: 'absolute',
                            top: '-40%',
                            right: '-25%',
                            width: '280px',
                            height: '280px',
                            borderRadius: '50%',
                            background: glowColor,
                            filter: 'blur(60px)',
                            pointerEvents: 'none',
                            zIndex: 0,
                        }}
                    />

                    {/* Top row: icon + badge */}
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', position: 'relative', zIndex: 1 }}>
                        {/* Icon */}
                        <motion.div
                            animate={{ rotate: hovered ? [0, -8, 8, 0] : 0, scale: hovered ? 1.08 : 1 }}
                            transition={spring(300, 14)}
                            style={{
                                width: '60px',
                                height: '60px',
                                borderRadius: '18px',
                                background: iconBg,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                flexShrink: 0,
                                boxShadow: `0 4px 16px ${accentColor}`,
                            }}
                        >
                            <Icon style={{ fontSize: 28, color: iconColor }} />
                        </motion.div>

                        {/* Badge */}
                        <div style={{
                            padding: '4px 12px',
                            borderRadius: '999px',
                            background: badgeColor,
                            color: badgeText,
                            fontSize: '11px',
                            fontWeight: 700,
                            fontFamily: "'Google Sans',sans-serif",
                            letterSpacing: '0.3px',
                            marginTop: '4px',
                        }}>
                            {badge}
                        </div>
                    </div>

                    {/* Text block */}
                    <div style={{ position: 'relative', zIndex: 1, flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{
                            fontFamily: "'Google Sans Display','Google Sans',sans-serif",
                            fontSize: '22px',
                            fontWeight: 600,
                            color: 'var(--md-on-surface)',
                            letterSpacing: '-0.4px',
                            lineHeight: 1.15,
                        }}>{title}</div>

                        <div style={{
                            fontFamily: "'Roboto Flex',sans-serif",
                            fontSize: '11px',
                            fontWeight: 600,
                            color: 'var(--md-primary)',
                            letterSpacing: '0.6px',
                            textTransform: 'uppercase',
                            opacity: 0.75,
                        }}>{subtitle}</div>

                        <div style={{
                            fontFamily: "'Roboto Flex',sans-serif",
                            fontSize: '13.5px',
                            fontWeight: 400,
                            color: 'var(--md-on-surface-variant)',
                            lineHeight: '1.6',
                            marginTop: '4px',
                        }}>{description}</div>

                        {/* Feature list */}
                        <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '7px' }}>
                            {features.map((feat, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: -8 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ ...spring(260, 22), delay: delay + 0.1 + i * 0.05 }}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'flex-start',
                                        gap: '8px',
                                        fontFamily: "'Roboto Flex',sans-serif",
                                        fontSize: '13px',
                                        color: 'var(--md-on-surface-variant)',
                                        lineHeight: 1.45,
                                    }}
                                >
                                    <span style={{
                                        width: '5px',
                                        height: '5px',
                                        borderRadius: '50%',
                                        background: 'var(--md-primary)',
                                        flexShrink: 0,
                                        marginTop: '6px',
                                        opacity: 0.7,
                                    }} />
                                    {feat}
                                </motion.div>
                            ))}
                        </div>
                    </div>

                    {/* CTA row */}
                    <div style={{ position: 'relative', zIndex: 1 }}>
                        <AnimatePresence>
                            <motion.div
                                animate={{ opacity: hovered ? 1 : 0.5, x: hovered ? 0 : -4 }}
                                transition={spring(340, 22)}
                                style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    padding: '8px 18px',
                                    borderRadius: '999px',
                                    background: hovered ? 'var(--md-primary)' : 'var(--md-primary-container)',
                                    color: hovered ? 'var(--md-on-primary)' : 'var(--md-on-primary-container)',
                                    fontSize: '13px',
                                    fontWeight: 600,
                                    fontFamily: "'Roboto Flex',sans-serif",
                                    transition: 'background 0.25s ease, color 0.25s ease',
                                }}
                            >
                                Open module
                                <ArrowForwardRoundedIcon style={{ fontSize: 15 }} />
                            </motion.div>
                        </AnimatePresence>
                    </div>
                </motion.div>
            </Link>
        </motion.div>
    )
}

/* ── Page ───────────────────────────────────────────────────── */
export default function HomePage() {
    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            gap: '20px',
            overflowY: 'auto',
            overflowX: 'hidden',
            position: 'relative',
            paddingBottom: '8px',
        }} className="scrollbar-none">


            {/* ══ HERO ═══════════════════════════════════════════════════ */}
            <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={spring(240, 26)}
                style={{
                    position: 'relative',
                    backgroundImage: 'radial-gradient(circle, rgb(255, 236, 210) 0%, rgb(165, 159, 252) 100%)',
                    backdropFilter: 'blur(24px)',
                    WebkitBackdropFilter: 'blur(24px)',
                    borderRadius: '32px',
                    padding: 'clamp(28px,4vw,48px) clamp(24px,4vw,52px)',
                    overflow: 'visible',
                    flexShrink: 0,
                    border: '1px solid rgba(196,199,197,0.30)',
                    boxShadow: '0 4px 32px rgba(0,0,0,0.04)',
                }}
            >
                {/* Hero blobs */}
                <div className="ambient-blob hero-blob-1" aria-hidden="true" />
                <div className="ambient-blob hero-blob-2" aria-hidden="true" />
                <div className="ambient-blob hero-blob-3" aria-hidden="true" />

                {/* Content */}
                <div style={{ position: 'relative', zIndex: 1 }}>

                    {/* Tagline chip */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.88, y: 8 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        transition={{ ...spring(320, 24), delay: 0.06 }}
                        style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '7px',
                            padding: '6px 16px',
                            background: 'rgba(255,255,255,0.7)',
                            backdropFilter: 'blur(8px)',
                            WebkitBackdropFilter: 'blur(8px)',
                            border: '1px solid rgba(196,199,197,0.5)',
                            color: 'var(--md-primary)',
                            borderRadius: '999px',
                            fontSize: '12px',
                            fontWeight: 700,
                            letterSpacing: '0.8px',
                            fontFamily: "'Google Sans',sans-serif",
                            marginBottom: '18px',
                            textTransform: 'uppercase',
                        }}
                    >
                        <span style={{
                            width: '7px', height: '7px', borderRadius: '50%',
                            background: 'var(--md-primary)', display: 'inline-block',
                            boxShadow: '0 0 8px rgba(11,87,208,0.6)',
                            animation: 'pulse-ring 1.4s cubic-bezier(0.4,0,0.6,1) infinite',
                        }} />
                        Contactless Health Screening
                    </motion.div>

                    {/* Headline */}
                    <motion.h1
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ ...spring(240, 26), delay: 0.12 }}
                        style={{
                            fontFamily: "'Playfair Display', serif",
                            fontSize: 'clamp(30px,3.8vw,52px)',
                            fontWeight: 500,
                            color: 'var(--md-on-surface)',
                            letterSpacing: '-1.5px',
                            lineHeight: 1.08,
                            marginBottom: '14px',
                            maxWidth: '700px',
                        }}
                    >
                        Health screening,{' '}
                        <span style={{ position: 'relative', display: 'inline-block' }}>
                            {/* Flower petal behind the word */}
                            <img
                                src={flowerPng}
                                alt=""
                                aria-hidden="true"
                                style={{
                                    position: 'absolute',
                                    right: '-52px',
                                    bottom: '-148px',
                                    width: '530px',
                                    height: '330px',
                                    objectFit: 'contain',
                                    opacity: 0.35,
                                    filter: 'hue-rotate(180deg) opacity(0.8) blur(0.5px)',
                                    transform: 'rotate(24deg) scale(1.1)',
                                    pointerEvents: 'none',
                                    zIndex: -1,
                                }}
                            />
                            <span style={{
                                position: 'relative',
                                zIndex: 1,
                                background: 'linear-gradient(135deg, var(--md-primary) 0%, #6750A4 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                backgroundClip: 'text',
                            }}>reimagined</span>
                        </span>{' '}
                        by AI.
                    </motion.h1>

                    {/* Subtitle */}
                    <motion.p
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ ...spring(240, 26), delay: 0.2 }}
                        style={{
                            fontFamily: "'Roboto Flex',sans-serif",
                            fontSize: 'clamp(14px,1.3vw,17px)',
                            fontWeight: 400,
                            color: 'var(--md-on-surface-variant)',
                            lineHeight: '1.65',
                            maxWidth: '520px',
                            marginBottom: '24px',
                        }}
                    >
                        Chiranjeevi uses camera, thermal, and mmWave radar sensors to assess
                        20+ biomarkers — instantly, non-invasively, and in your language.
                    </motion.p>

                    {/* Feature tags */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5, duration: 0.3 }}
                        style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '28px' }}
                    >
                        {TAGS.map((t, i) => (
                            <FeatureTag key={t.label} Icon={t.icon} label={t.label} index={i} />
                        ))}
                    </motion.div>

                    {/* Stats row */}
                    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                        {STATS.map((s, i) => (
                            <StatCard key={s.label} {...s} index={i} />
                        ))}
                    </div>
                </div>
            </motion.div>

            {/* ══ SECTION LABEL ══════════════════════════════════════════ */}
            <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ ...spring(260, 24), delay: 0.45 }}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    flexShrink: 0,
                }}
            >
                <div style={{
                    fontFamily: "'Google Sans',sans-serif",
                    fontSize: '13px',
                    fontWeight: 700,
                    letterSpacing: '1px',
                    textTransform: 'uppercase',
                    color: 'var(--md-on-surface-variant)',
                }}>Explore modules</div>
                <div style={{ flex: 1, height: '1px', background: 'var(--md-outline-variant)', opacity: 0.6 }} />
            </motion.div>

            {/* ══ MODULE CARDS ═══════════════════════════════════════════ */}
            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(290px, 1fr))',
                    gap: '20px',
                    flexShrink: 0,
                }}
            >
                {MODULES.map((mod) => (
                    <ModuleCard key={mod.to} {...mod} />
                ))}
            </div>
        </div>
    )
}

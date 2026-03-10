import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded'
import { useState } from 'react'

const cardSpring = { type: 'spring', stiffness: 300, damping: 20 }
const arrowSpring = { type: 'spring', stiffness: 380, damping: 24 }

/**
 * ModuleCard — M3 Expressive asymmetric card linking to an app module.
 *
 * @param {object}   props
 * @param {React.ElementType} props.IconComponent  MUI icon component
 * @param {string}   props.iconBg      CSS color for icon container bg
 * @param {string}   props.iconColor   CSS color for icon itself
 * @param {string}   props.title       Card title
 * @param {string}   props.description Short description
 * @param {string}   props.to          React Router link target
 * @param {string}   props.borderRadius Asymmetric border-radius value
 * @param {number}   props.delay       Framer Motion entrance delay (seconds)
 * @param {string}   props.accentColor Shadow/glow accent (rgba)
 */
export default function ModuleCard({
    // eslint-disable-next-line no-unused-vars
    IconComponent,
    iconBg,
    iconColor,
    title,
    description,
    to,
    borderRadius = '28px 28px 12px 28px',
    delay = 0.15,
    accentColor = 'rgba(11, 87, 208, 0.16)',
}) {
    const [hovered, setHovered] = useState(false)

    return (
        <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...cardSpring, delay }}
        >
            <Link to={to} style={{ textDecoration: 'none', display: 'block' }}>
                <motion.div
                    onHoverStart={() => setHovered(true)}
                    onHoverEnd={() => setHovered(false)}
                    whileHover={{
                        y: -7,
                        scale: 1.02,
                        boxShadow: `0 16px 40px ${accentColor}`,
                    }}
                    whileTap={{ scale: 0.98 }}
                    transition={cardSpring}
                    style={{
                        background: 'var(--md-surface-container)',
                        borderRadius,
                        padding: '28px',
                        cursor: 'pointer',
                        position: 'relative',
                        overflow: 'hidden',
                        boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
                        minHeight: '200px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '16px',
                    }}
                >
                    {/* ── Decorative tonal blob in card background ── */}
                    <motion.div
                        animate={{
                            opacity: hovered ? 0.6 : 0.3,
                            scale: hovered ? 1.1 : 1,
                        }}
                        transition={{ type: 'spring', stiffness: 160, damping: 22 }}
                        style={{
                            position: 'absolute',
                            top: '-30%',
                            right: '-20%',
                            width: '200px',
                            height: '200px',
                            borderRadius: '50%',
                            background: iconBg,
                            filter: 'blur(48px)',
                            pointerEvents: 'none',
                            zIndex: 0,
                        }}
                    />

                    {/* ── Icon container ── */}
                    <motion.div
                        animate={{ rotate: hovered ? [0, -6, 6, 0] : 0 }}
                        transition={{ type: 'spring', stiffness: 300, damping: 12 }}
                        style={{
                            width: '64px',
                            height: '64px',
                            borderRadius: '20px',
                            background: iconBg,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            position: 'relative',
                            zIndex: 1,
                        }}
                    >
                        <IconComponent style={{ fontSize: 30, color: iconColor }} />
                    </motion.div>

                    {/* ── Text block ── */}
                    <div style={{ flex: 1, position: 'relative', zIndex: 1 }}>
                        <div
                            style={{
                                fontFamily: "'Google Sans Display', 'Google Sans', sans-serif",
                                fontSize: '20px',
                                fontWeight: 500,
                                color: 'var(--md-on-surface)',
                                marginBottom: '8px',
                                letterSpacing: '-0.2px',
                            }}
                        >
                            {title}
                        </div>
                        <div
                            style={{
                                fontFamily: "'Roboto Flex', sans-serif",
                                fontSize: '14px',
                                fontWeight: 400,
                                color: 'var(--md-on-surface-variant)',
                                lineHeight: '1.55',
                            }}
                        >
                            {description}
                        </div>
                    </div>

                    {/* ── Arrow chip (appears on hover) ── */}
                    <AnimatePresence>
                        {hovered && (
                            <motion.div
                                initial={{ x: -8, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                exit={{ x: -8, opacity: 0 }}
                                transition={arrowSpring}
                                style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    padding: '6px 14px',
                                    borderRadius: '999px',
                                    background: 'var(--md-primary-container)',
                                    color: 'var(--md-on-primary-container)',
                                    fontSize: '13px',
                                    fontWeight: 600,
                                    alignSelf: 'flex-start',
                                    position: 'relative',
                                    zIndex: 1,
                                }}
                            >
                                Open
                                <ArrowForwardRoundedIcon style={{ fontSize: 16 }} />
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>
            </Link>
        </motion.div>
    )
}

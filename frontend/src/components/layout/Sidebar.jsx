import { useLocation, NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import HomeRoundedIcon from '@mui/icons-material/HomeRounded'
import BiotechRoundedIcon from '@mui/icons-material/BiotechRounded'
import SmartToyRoundedIcon from '@mui/icons-material/SmartToyRounded'
import heartIcon from '../../assets/heart_icon.png'
import LanguageSwitcher from './LanguageSwitcher'
/* ── Nav item definitions ─────────────────────────────────────── */
const NAV_ITEMS = [
    { to: '/', labelKey: 'sidebar.home', icon: HomeRoundedIcon },
    { to: '/screening', labelKey: 'sidebar.screening', icon: BiotechRoundedIcon },
    { to: '/chatbot', labelKey: 'sidebar.chatbot', icon: SmartToyRoundedIcon },
]

/* ── Spring configs ───────────────────────────────────────────── */
const sidebarSpring = { type: 'spring', stiffness: 280, damping: 26 }
const itemSpring = { type: 'spring', stiffness: 340, damping: 24 }
const pillSpring = { type: 'spring', stiffness: 380, damping: 30 }

/* ── Individual Nav Item ─────────────────────────────────────── */
function NavItem({ item }) {
    const location = useLocation()
    const { t } = useTranslation()

    const Icon = item.icon
    const label = t(item.labelKey)

    return (
        <NavLink
            to={item.to}
            end={item.to === '/'}
            className="relative block focus-visible:outline-none"
            aria-label={label}
        >
            {() => {
                const active = item.to === '/'
                    ? location.pathname === '/'
                    : location.pathname.startsWith(item.to)

                return (
                    <motion.div
                        className="relative flex items-center gap-4 px-5 py-3 rounded-full
                       min-h-[48px] cursor-pointer select-none touch-target
                       text-sm font-medium leading-none"
                        style={{
                            color: active
                                ? 'var(--md-on-primary-container)'
                                : 'var(--md-on-surface-variant)',
                        }}
                        whileHover={{ x: active ? 0 : 3, scale: 1.01 }}
                        whileTap={{ scale: 0.97 }}
                        transition={itemSpring}
                    >
                        {/* ── Sliding active background ── */}
                        {active && (
                            <motion.div
                                layoutId="nav-active-pill"
                                className="absolute inset-0 rounded-full"
                                style={{ background: 'var(--md-primary-container)' }}
                                transition={pillSpring}
                            />
                        )}

                        {/* ── Hover tint (inactive only) ── */}
                        {!active && (
                            <motion.div
                                className="absolute inset-0 rounded-full opacity-0"
                                style={{ background: 'var(--md-surface-container-high)' }}
                                whileHover={{ opacity: 1 }}
                                transition={{ duration: 0.15 }}
                            />
                        )}

                        {/* ── Icon ── */}
                        <span className="relative z-10 flex-shrink-0 flex items-center justify-center w-6 h-6">
                            <Icon
                                style={{
                                    fontSize: 22,
                                    fontVariationSettings: active ? "'FILL' 1" : "'FILL' 0",
                                    transition: 'font-variation-settings 200ms ease',
                                }}
                            />
                        </span>

                        {/* ── Label ── */}
                        <span
                            className="relative z-10 font-body truncate"
                            style={{
                                fontFamily: "'Roboto Flex', sans-serif",
                                fontWeight: active ? 600 : 500,
                                fontSize: '14px',
                                letterSpacing: '0.1px',
                            }}
                        >
                            {label}
                        </span>
                    </motion.div>
                )
            }}
        </NavLink>
    )
}

/* ── Sidebar ─────────────────────────────────────────────────── */
export default function Sidebar() {
    return (
        <motion.aside
            id="app-sidebar"
            initial={{ x: -40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={sidebarSpring}
            className="flex flex-col flex-shrink-0 overflow-hidden"
            style={{
                width: 'var(--sidebar-width)',
                height: '100%',
                background: 'rgba(240, 237, 244, 0.75)',
                backdropFilter: 'blur(24px) saturate(1.5)',
                WebkitBackdropFilter: 'blur(24px) saturate(1.5)',
                borderRight: '1px solid rgba(196, 199, 197, 0.5)',
                padding: '20px 12px',
                zIndex: 20,
            }}
        >
            {/* ── Logo ── */}
            <motion.div
                className="flex items-center gap-2 px-3 mb-6 flex-shrink-0"
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ ...sidebarSpring, delay: 0.08 }}
            >
                <motion.div
                    whileHover={{ rotate: [0, -8, 8, 0] }}
                    transition={{ type: 'spring', stiffness: 400, damping: 10 }}
                >
                    <img
                        src={heartIcon}
                        alt="Heart Icon"
                        style={{
                            width: 30,
                            height: 30,
                            objectFit: 'contain',
                        }}
                    />
                </motion.div>
                <span
                    style={{
                        fontFamily: "'Google Sans Display', 'Google Sans', sans-serif",
                        fontSize: '20px',
                        fontWeight: 300,
                        color: 'var(--md-primary)',
                        letterSpacing: '-0.3px',
                    }}
                >
                    Chiranjeevi
                </span>
            </motion.div>

            {/* ── Nav Menu ── */}
            <nav
                className="flex flex-col gap-1 flex-1"
                role="navigation"
                aria-label="Main navigation"
            >
                {NAV_ITEMS.map((item, i) => (
                    <motion.div
                        key={item.to}
                        initial={{ opacity: 0, x: -16 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ ...sidebarSpring, delay: 0.1 + i * 0.06 }}
                    >
                        <NavItem item={item} />
                    </motion.div>
                ))}
            </nav>

            {/* ── Footer ── */}
            <div className="mt-auto pt-4 flex flex-col gap-4">
                <LanguageSwitcher />

                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.35, duration: 0.4 }}
                    style={{
                        padding: '12px 8px 0',
                        borderTop: '1px solid var(--md-surface-container-high)',
                        fontSize: '11px',
                        fontWeight: 500,
                        letterSpacing: '0.4px',
                        color: 'var(--md-outline)',
                        fontFamily: "'Roboto Flex', sans-serif",
                        textAlign: 'center',
                    }}
                >
                    v3.0 &bull; M3 Expressive
                </motion.div>
            </div>
        </motion.aside>
    )
}

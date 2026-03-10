import { useLocation, Outlet } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './Sidebar'

const pageVariants = {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -8 },
}

const pageTransition = {
    type: 'spring',
    stiffness: 300,
    damping: 30,
    mass: 0.7,
}

export default function AppShell() {
    const location = useLocation()

    return (
        /*
         * Root layout: sidebar (fixed width) + scrollable main.
         * Responsive: sidebar moves to bottom on ≤900px via CSS.
         */
        <div className="flex fill-parent overflow-hidden" id="app-shell" style={{ position: 'relative' }}>

            {/* ══ Ambient Mesh Gradient Blobs — global backdrop ══ */}
            <div className="ambient-blob ambient-blob-1" aria-hidden="true" style={{ zIndex: -1 }} />
            <div className="ambient-blob ambient-blob-2" aria-hidden="true" style={{ zIndex: -1 }} />
            <div className="ambient-blob ambient-blob-3" aria-hidden="true" style={{ zIndex: -1 }} />

            {/* ── Top-right corner glow ── */}
            <div className="corner-glow-tr" aria-hidden="true" />

            <Sidebar />

            {/* ── Scroll Fade Overlays ── */}
            <div
                aria-hidden="true"
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 'var(--sidebar-width)',
                    right: 0,
                    height: '48px',
                    pointerEvents: 'none',
                    zIndex: 10,
                    background: 'linear-gradient(to bottom, rgba(253,251,255,0.97) 0%, rgba(253,251,255,0) 100%)',
                }}
            />
            <div
                aria-hidden="true"
                style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 'var(--sidebar-width)',
                    right: 0,
                    height: '48px',
                    pointerEvents: 'none',
                    zIndex: 10,
                    background: 'linear-gradient(to top, rgba(253,251,255,0.97) 0%, rgba(253,251,255,0) 100%)',
                }}
            />

            {/* ── Home-only vignette overlay (must live outside motion.main to avoid filter stacking context) ── */}
            {location.pathname === '/' && (
                <div aria-hidden="true" style={{
                    position: 'absolute',
                    top: 0,
                    right: 0,
                    bottom: 0,
                    left: 0,
                    pointerEvents: 'none',
                    zIndex: 9,
                    background: [
                        'radial-gradient(ellipse 55% 38% at 0% 0%,   rgba(253,251,255,0.82) 0%, transparent 100%)',
                        'radial-gradient(ellipse 55% 38% at 100% 0%,  rgba(253,251,255,0.82) 0%, transparent 100%)',
                        'radial-gradient(ellipse 55% 38% at 0% 100%, rgba(253,251,255,0.82) 0%, transparent 100%)',
                        'radial-gradient(ellipse 55% 38% at 100% 100%,rgba(253,251,255,0.82) 0%, transparent 100%)',
                    ].join(','),
                }} />
            )}

            {/* ── Main content area ── */}
            <AnimatePresence mode="wait">
                <motion.main
                    key={location.pathname}
                    variants={pageVariants}
                    initial="initial"
                    animate="animate"
                    exit="exit"
                    transition={pageTransition}
                    className="flex flex-col flex-1 min-w-0 overflow-hidden"
                    style={{ background: 'transparent', padding: '16px' }}
                >
                    <Outlet />
                </motion.main>
            </AnimatePresence>
        </div>
    )
}

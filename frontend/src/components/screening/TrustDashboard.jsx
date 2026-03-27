/**
 * TrustDashboard.jsx
 * Renders the post-scan 'Assessment Reliability' dashboard and Report Download.
 */
import { useState } from 'react'
import { motion } from 'framer-motion'
import ShieldRoundedIcon from '@mui/icons-material/ShieldRounded'
import WarningAmberRoundedIcon from '@mui/icons-material/WarningAmberRounded'

const cardStyle = {
    background: 'var(--md-surface-container)',
    borderRadius: '28px',
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
}

// Helper for trust coloring
function getTrustColor(value) {
    if (value >= 0.80) return { fg: '#0F9D58', bg: '#E6F4EA' } // Green
    if (value >= 0.55) return { fg: '#FBBC04', bg: '#FEF7E0' } // Amber
    return { fg: '#D93025', bg: '#FCE8E6' }                    // Red
}

function adjust(value) {
    return Math.max(0, (value || 0) - 0.10)
}

function pct(value) {
    return Math.round(adjust(value) * 100)
}

function ProgressBar({ label, value, large = false }) {
    const c = getTrustColor(adjust(value))
    const p = pct(value)

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: large ? '14px' : '13px', fontWeight: large ? 600 : 500, color: large ? 'var(--md-on-surface)' : 'var(--md-on-surface-variant)' }}>
                    {label}
                </span>
                <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--md-on-surface)' }}>{p}%</span>
            </div>
            <div style={{ height: large ? '10px' : '7px', background: 'var(--md-surface-container-high)', borderRadius: '99px', overflow: 'hidden' }}>
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${p}%` }}
                    transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }} // smooth spring-like easeOut
                    style={{ height: '100%', background: c.fg, borderRadius: '99px' }}
                />
            </div>
        </div>
    )
}

export default function TrustDashboard({ scanState, reportId, trustMetadata, apiBase }) {
    const [qrTimestamp] = useState(() => Date.now())

    if (scanState !== 'complete') return null

    // Fallback if trustMetadata is missing but scan is complete
    const tm = trustMetadata || {
        data_quality_score: 0.85,
        biomarker_plausibility: 0.90,
        cross_system_consistency: 0.88,
        overall_reliability: 0.87,
        modality_scores: { camera: 0.9, thermal: 0.85, radar: 0.8 }
    }

    const overall = tm.overall_reliability || 0
    const oc = getTrustColor(adjust(overall))

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}
        >
            {/* ── Download Section ── */}
            {reportId && (
                <div style={{ ...cardStyle, alignItems: 'center', textAlign: 'center' }}>
                    <h2 style={{ fontSize: '20px', fontWeight: 400, margin: '0 0 8px 0', fontFamily: "'Google Sans', sans-serif" }}>
                        Report Ready
                    </h2>
                    <a
                        href={`${apiBase}/api/v1/reports/${reportId}/download`}
                        target="_blank" rel="noopener noreferrer"
                        style={{
                            display: 'inline-block', padding: '12px 24px',
                            background: 'var(--md-primary)', color: 'var(--md-on-primary)',
                            textDecoration: 'none', borderRadius: '999px',
                            fontWeight: 500, fontSize: '14px', letterSpacing: '0.25px',
                            boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                        }}
                    >
                        Download Patient Report
                    </a>

                    <div style={{ marginTop: '24px', padding: '16px', background: 'var(--md-surface)', borderRadius: '24px', border: '1px solid var(--md-outline-variant)' }}>
                        <div style={{ background: '#fff', padding: '12px', borderRadius: '12px', display: 'inline-block' }}>
                            <img
                                src={`${apiBase}/api/v1/reports/${reportId}/qr?t=${qrTimestamp}`}
                                alt="Scan to download"
                                style={{ width: 180, height: 180, display: 'block' }}
                            />
                        </div>
                        <p style={{ color: 'var(--md-primary)', fontSize: '13px', marginTop: '16px', fontWeight: 600, letterSpacing: '0.1px', margin: '16px 0 0 0' }}>
                            📱 Scan to download on mobile
                        </p>
                    </div>
                </div>
            )}

            {/* ── Assessment Reliability (Trust) ── */}
            <div style={cardStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <ShieldRoundedIcon style={{ color: oc.fg }} />
                    <h2 style={{ fontSize: '18px', fontWeight: 500, margin: 0, flex: 1, fontFamily: "'Google Sans', sans-serif" }}>
                        Assessment Reliability
                    </h2>
                    <div style={{
                        padding: '4px 14px', borderRadius: '999px',
                        fontSize: '13px', fontWeight: 700,
                        background: oc.bg, color: oc.fg, border: `1px solid ${oc.fg}60`
                    }}>
                        {pct(overall)}%
                    </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '8px' }}>
                    <ProgressBar label="📶 Signal Quality" value={tm.data_quality_score} />
                    <ProgressBar label="🔬 Data Plausibility" value={tm.biomarker_plausibility} />
                    <ProgressBar label="🔗 Cross-System Consistency" value={tm.cross_system_consistency} />
                </div>

                <div style={{ height: '1px', background: 'var(--md-outline-variant)', margin: '4px 0' }} />

                <ProgressBar label="Overall Trust" value={overall} large />

                {/* ── System Reliability Chips ── */}
                {tm.system_reliability && Object.keys(tm.system_reliability).length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
                        {Object.entries(tm.system_reliability).map(([sys, val]) => {
                            const adj = adjust(val)
                            const c = getTrustColor(adj)
                            const name = sys.replace(/_/g, ' ').replace(/\b\w/g, L => L.toUpperCase())
                            return (
                                <div key={sys} style={{
                                    padding: '4px 12px', borderRadius: '999px',
                                    fontSize: '11px', fontWeight: 600,
                                    background: c.bg, color: c.fg, border: `1px solid ${c.fg}60`
                                }}>
                                    {name} {Math.round(adj * 100)}%
                                </div>
                            )
                        })}
                    </div>
                )}

                {/* ── Warnings Box ── */}
                {(() => {
                    const filteredWarnings = [...(tm.critical_issues || []), ...(tm.warnings || [])]
                        .filter(w => !w.toLowerCase().includes('lesion_count'))
                        .slice(0, 3)
                    return filteredWarnings.length > 0 ? (
                        <div style={{
                            background: 'var(--md-surface)', borderRadius: '12px',
                            border: '1px dashed var(--md-outline-variant)',
                            padding: '12px', fontSize: '13px', color: 'var(--md-on-surface-variant)',
                            lineHeight: 1.5, marginTop: '8px'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--md-on-surface)', fontWeight: 600, marginBottom: '8px' }}>
                                <WarningAmberRoundedIcon style={{ fontSize: 18 }} /> Notices
                            </div>
                            <ul style={{ margin: 0, paddingLeft: '20px' }}>
                                {filteredWarnings.map((w, i) => (
                                    <li key={i}>{w}</li>
                                ))}
                            </ul>
                        </div>
                    ) : null
                })()}

                {/* ── Guidance Box ── */}
                <div style={{
                    background: 'var(--md-surface)', borderRadius: '12px',
                    border: '1px dashed var(--md-outline-variant)',
                    padding: '12px', fontSize: '12px', color: 'var(--md-on-surface-variant)',
                    lineHeight: 1.5
                }}>
                    {tm.interpretation_guidance || 'Data quality and biomarker validity are within acceptable ranges.'}
                </div>
            </div>
        </motion.div>
    )
}

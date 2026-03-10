/**
 * MessageBubble.jsx
 * Renders a single chat message — user or assistant.
 * Assistant messages stream tokens then render final markdown.
 */
import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { marked } from 'marked'
import CitationPill from './CitationPill'

// Configure marked for safe inline rendering
marked.setOptions({ breaks: true, gfm: true })

const bubbleSpring = { type: 'spring', stiffness: 280, damping: 24 }

export default function MessageBubble({ message, onCitationClick }) {
    const { role, content, isStreaming, renderMarkdown, citations } = message
    const isUser = role === 'user'
    const contentRef = useRef(null)

    // Auto-scroll to keep cursor visible while streaming
    useEffect(() => {
        if (isStreaming) {
            contentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
        }
    }, [content, isStreaming])

    const htmlContent = renderMarkdown && content
        ? marked.parse(content)
        : null

    return (
        <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={bubbleSpring}
            style={{
                display: 'flex',
                justifyContent: isUser ? 'flex-end' : 'flex-start',
                padding: '4px 0',
            }}
        >
            <div style={{ maxWidth: '78%', display: 'flex', flexDirection: 'column' }}>
                <div
                    ref={contentRef}
                    style={{
                        padding: isUser ? '12px 18px' : '14px 18px',
                        borderRadius: isUser
                            ? '20px 6px 20px 20px'
                            : '6px 20px 20px 20px',
                        background: isUser
                            ? 'var(--md-primary-container)'
                            : 'var(--md-surface-container)',
                        color: isUser
                            ? 'var(--md-on-primary-container)'
                            : 'var(--md-on-surface)',
                        fontFamily: "'Google Sans', 'Roboto Flex', sans-serif",
                        fontSize: '14.5px',
                        lineHeight: '1.6',
                        position: 'relative',
                        boxShadow: isUser
                            ? '0 1px 4px rgba(11,87,208,0.12)'
                            : '0 1px 4px rgba(0,0,0,0.06)',
                    }}
                >
                    {/* Streaming cursor */}
                    {isStreaming && (
                        <motion.span
                            animate={{ opacity: [1, 0, 1] }}
                            transition={{ duration: 0.7, repeat: Infinity }}
                            style={{
                                display: 'inline-block', width: '2px', height: '14px',
                                background: 'var(--md-primary)', borderRadius: '1px',
                                marginLeft: '2px', verticalAlign: 'middle',
                            }}
                        />
                    )}

                    {/* Rendered content */}
                    {renderMarkdown && htmlContent ? (
                        <div
                            className="prose-m3"
                            style={{ fontFamily: "'Google Sans', 'Roboto Flex', sans-serif" }}
                            dangerouslySetInnerHTML={{ __html: htmlContent }}
                        />
                    ) : (
                        <span style={{ whiteSpace: 'pre-wrap' }}>{content}</span>
                    )}
                </div>

                {/* Citation pill — assistant only */}
                {!isUser && citations?.length > 0 && (
                    <div style={{ marginTop: '4px', paddingLeft: '4px' }}>
                        <CitationPill
                            citations={citations}
                            onClick={() => onCitationClick?.(citations)}
                        />
                    </div>
                )}
            </div>
        </motion.div>
    )
}

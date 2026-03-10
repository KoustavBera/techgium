/**
 * MessageList.jsx
 * Scrollable message container — auto-scrolls to bottom on new messages.
 */
import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function MessageList({ messages, onCitationClick }) {
    const bottomRef = useRef(null)

    // Auto-scroll to newest message
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages.length])

    if (messages.length === 0) {
        return (
            <div style={{
                flex: 1, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', gap: '12px',
                color: 'var(--md-on-surface-variant)',
                fontFamily: "'Google Sans', sans-serif",
            }}>
                <span style={{ fontSize: '44px' }}>🩺</span>
                <div style={{ fontSize: '16px', fontWeight: 500, color: 'var(--md-on-surface)' }}>
                    Ask Dr. Chiranjeevi anything
                </div>
                <div style={{ fontSize: '13px', maxWidth: '300px', textAlign: 'center', lineHeight: 1.5 }}>
                    Visit symptoms, medications, lab values, report summaries — in English, Hindi, Bengali or Kannada.
                </div>
            </div>
        )
    }

    return (
        <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '16px 20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
        }}>
            {messages.map(msg => (
                <MessageBubble
                    key={msg.id}
                    message={msg}
                    onCitationClick={onCitationClick}
                />
            ))}
            <div ref={bottomRef} />
        </div>
    )
}

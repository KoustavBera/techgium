/**
 * ChatInput.jsx
 * Auto-expanding textarea with mic button and send button.
 */
import { useState, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import SendRoundedIcon from '@mui/icons-material/SendRounded'
import MicRoundedIcon from '@mui/icons-material/MicRounded'
import StopRoundedIcon from '@mui/icons-material/StopRounded'

const btnSpring = { type: 'spring', stiffness: 380, damping: 22 }

// Stable animation config — defined outside component to prevent re-registration on re-render
const glowAnimate = {
    boxShadow: [
        '0 0 0px 0px rgba(11,87,208,0)',
        '0 0 28px 6px rgba(131,168,255,0.7)',
        '0 0 0px 0px rgba(11,87,208,0)',
    ],
}
const glowTransition = { duration: 3, ease: 'easeInOut', repeat: Infinity, repeatType: 'loop' }


export default function ChatInput({
    onSend,
    isProcessing,
    isListening,
    onMicClick,
    isMicSupported,
}) {
    const [text, setText] = useState('')
    const textareaRef = useRef(null)

    // Auto-resize textarea up to 120px
    const handleInput = useCallback((e) => {
        const el = e.target
        el.style.height = 'auto'
        el.style.height = `${Math.min(el.scrollHeight, 120)}px`
        setText(el.value)
    }, [])

    const handleSend = useCallback(() => {
        const trimmed = text.trim()
        if (!trimmed || isProcessing) return
        onSend(trimmed)
        setText('')
        if (textareaRef.current) {
            textareaRef.current.value = ''
            textareaRef.current.style.height = 'auto'
        }
    }, [text, isProcessing, onSend])

    const handleKeyDown = useCallback((e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }, [handleSend])

    const canSend = text.trim().length > 0 && !isProcessing

    return (
        <div style={{
            padding: '12px 16px 16px',
            background: '#ffff',
            borderRadius: '0 0 24px 24px',
            borderTop: '1px solid var(--md-surface-container-high)',
            flexShrink: 0,
        }}>
            <motion.div
                initial={{ boxShadow: '0 0 0px 0px rgba(11,87,208,0)' }}
                animate={glowAnimate}
                transition={glowTransition}
                style={{
                    display: 'flex',
                    alignItems: 'flex-end',
                    gap: '10px',
                    background: '#edededff',
                    borderRadius: '999px',
                    border: '1px solid #433d7aff',
                    padding: '8px 8px 8px 20px',
                }}
            >
                {/* Textarea */}
                <textarea
                    ref={textareaRef}
                    rows={1}
                    placeholder={isProcessing ? 'Waiting for response…' : 'Ask Dr. Chiranjeevi…'}
                    disabled={isProcessing}
                    onInput={handleInput}
                    onKeyDown={handleKeyDown}
                    style={{
                        flex: 1,
                        border: 'none',
                        outline: 'none',
                        background: 'transparent',
                        resize: 'none',
                        fontFamily: "'Google Sans', 'Roboto Flex', sans-serif",
                        fontSize: '14.5px',
                        lineHeight: '1.5',
                        color: 'var(--md-on-surface)',
                        padding: '4px 0',
                        overflowY: 'auto',
                        maxHeight: '120px',
                    }}
                />

                {/* Mic button */}
                {isMicSupported && (
                    <motion.button
                        whileHover={{ scale: 1.08 }}
                        whileTap={{ scale: 0.93 }}
                        transition={btnSpring}
                        onClick={onMicClick}
                        aria-label={isListening ? 'Stop recording' : 'Start voice input'}
                        style={{
                            width: 40, height: 40,
                            borderRadius: '50%',
                            border: 'none',
                            cursor: 'pointer',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            background: isListening
                                ? 'var(--md-error)'
                                : 'var(--md-surface-container-high)',
                            color: isListening
                                ? 'var(--md-on-error)'
                                : 'var(--md-on-surface-variant)',
                            flexShrink: 0,
                            boxShadow: isListening ? '0 0 0 6px rgba(179,38,30,0.15)' : 'none',
                            transition: 'box-shadow 0.2s ease, background 0.2s ease',
                        }}
                    >
                        {isListening
                            ? <StopRoundedIcon style={{ fontSize: 20 }} />
                            : <MicRoundedIcon style={{ fontSize: 20 }} />
                        }
                    </motion.button>
                )}

                {/* Send button */}
                <motion.button
                    whileHover={canSend ? { scale: 1.08 } : {}}
                    whileTap={canSend ? { scale: 0.93 } : {}}
                    transition={btnSpring}
                    onClick={handleSend}
                    disabled={!canSend}
                    aria-label="Send message"
                    style={{
                        width: 40, height: 40,
                        borderRadius: '50%',
                        border: 'none',
                        cursor: canSend ? 'pointer' : 'default',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        background: canSend
                            ? 'var(--md-primary)'
                            : 'var(--md-surface-container-highest)',
                        color: canSend
                            ? 'var(--md-on-primary)'
                            : 'var(--md-on-surface-variant)',
                        flexShrink: 0,
                        opacity: canSend ? 1 : 0.5,
                        transition: 'background 0.2s ease, opacity 0.2s ease',
                    }}
                >
                    <SendRoundedIcon style={{ fontSize: 18 }} />
                </motion.button>
            </motion.div>

            {/* Footer hint */}
            <div style={{
                textAlign: 'center', marginTop: '8px',
                fontSize: '11px', color: 'var(--md-outline)',
                fontFamily: "'Google Sans', sans-serif",
            }}>
                AI responses are for informational purposes only — not medical advice
            </div>
        </div >
    )
}

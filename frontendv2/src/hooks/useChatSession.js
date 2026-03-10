/**
 * useChatSession.js
 * Manages the full chatbot session: message state, SSE streaming,
 * token accumulation, translation, audio TTS, and citations.
 */
import { useState, useRef, useCallback } from 'react'
import { streamFetch, checkHealth } from '../lib/api'

const AGENT_STATE = {
    ANALYZING: 'analyzing',
    SEARCHING_WEB: 'searching_web',
    SEARCHING_PUBMED: 'searching_pubmed',
    THINKING: 'thinking',
    STREAMING: 'streaming',
}

const STATE_LABELS = {
    [AGENT_STATE.ANALYZING]: '🔀 Analyzing your question',
    [AGENT_STATE.SEARCHING_WEB]: '🌐 Searching the web',
    [AGENT_STATE.SEARCHING_PUBMED]: '📄 Searching medical literature',
    [AGENT_STATE.THINKING]: '🩺 Dr. Chiranjeevi is thinking',
    [AGENT_STATE.STREAMING]: '💬 Responding',
}

function makeId() {
    return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
}

export function useChatSession() {
    const [messages, setMessages] = useState([])
    const [isProcessing, setIsProcessing] = useState(false)
    const [typingState, setTypingState] = useState(null)  // null = hidden
    const [connectionStatus, setConnectionStatus] = useState('checking')
    const audioRef = useRef(new Audio())

    // ── Check backend health ──────────────────────────────────────
    const checkConnection = useCallback(async () => {
        const ok = await checkHealth()
        setConnectionStatus(ok ? 'connected' : 'disconnected')
    }, [])

    // ── Add a message to the list ─────────────────────────────────
    const addMessage = useCallback((role, content = '', extra = {}) => {
        const id = makeId()
        setMessages(prev => [...prev, { id, role, content, ...extra }])
        return id
    }, [])

    // ── Update a specific message by id ──────────────────────────
    const updateMessage = useCallback((id, patch) => {
        setMessages(prev =>
            prev.map(m => m.id === id ? { ...m, ...patch } : m)
        )
    }, [])

    // ── Main send handler ─────────────────────────────────────────
    const sendMessage = useCallback(async (text, language = 'en-IN') => {
        if (!text.trim() || isProcessing) return

        setIsProcessing(true)

        // Add user bubble immediately
        addMessage('user', text)

        // Placeholder assistant bubble
        const assistantId = addMessage('assistant', '', { isStreaming: true })

        let accumulatedEnglish = ''
        let translationApplied = false

        try {
            const response = await streamFetch('/api/v1/doctor/chat', {
                query: text,
                patient_id: 'WEB_USER',
                language,
            })

            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() // keep incomplete chunk

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue
                    try {
                        const data = JSON.parse(line.slice(6))

                        // ── Status / typing indicator ──
                        if (data.type === 'status') {
                            if (data.stage === 'citations') {
                                // Citation payload
                                try {
                                    const parsed = JSON.parse(data.message)
                                    if (Array.isArray(parsed) && parsed.length > 0) {
                                        updateMessage(assistantId, { citations: parsed })
                                    }
                                } catch { /* ignore */ }
                            } else {
                                setTypingState(
                                    STATE_LABELS[data.stage] || data.message || 'Processing…'
                                )
                            }

                            // ── Streaming token ──
                        } else if (data.type === 'token') {
                            accumulatedEnglish += data.token
                            updateMessage(assistantId, {
                                content: language === 'en-IN'
                                    ? accumulatedEnglish
                                    : '⏳ Translating to your language…',
                                isStreaming: true,
                            })

                            // ── Final translated payload ──
                        } else if (data.type === 'final_translated') {
                            translationApplied = true
                            updateMessage(assistantId, {
                                content: data.text,
                                isStreaming: false,
                                renderMarkdown: true,
                            })
                            // TTS audio
                            if (data.audio_base64) {
                                audioRef.current.src = `data:audio/wav;base64,${data.audio_base64}`
                                audioRef.current.play().catch(() => { })
                            }
                        }
                    } catch { /* malformed chunk, skip */ }
                }
            }

            // Finalise English session — render as markdown
            if (!translationApplied) {
                updateMessage(assistantId, {
                    content: accumulatedEnglish,
                    isStreaming: false,
                    renderMarkdown: true,
                })
            }

        } catch (err) {
            updateMessage(assistantId, {
                content: `⚠️ Error: ${err.message}. Make sure the backend is running at http://localhost:8000.`,
                isStreaming: false,
            })
        } finally {
            setIsProcessing(false)
            setTypingState(null)
        }
    }, [isProcessing, addMessage, updateMessage])

    // ── Clear conversation ────────────────────────────────────────
    const clearMessages = useCallback(() => setMessages([]), [])

    return {
        messages,
        isProcessing,
        typingState,
        connectionStatus,
        sendMessage,
        clearMessages,
        checkConnection,
    }
}

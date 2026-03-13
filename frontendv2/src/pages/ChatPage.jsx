/**
 * ChatPage.jsx
 * Assembles the full CHIRANJEEVAI chatbot interface.
 */
import { useState, useEffect } from 'react'
import { useChatSession } from '../hooks/useChatSession'
import { useSpeechRecognition } from '../hooks/useSpeechRecognition'
import LanguageOverlay from '../components/chat/LanguageOverlay'
import ChatHeader from '../components/chat/ChatHeader'
import MessageList from '../components/chat/MessageList'
import TypingIndicator from '../components/chat/TypingIndicator'
import ChatInput from '../components/chat/ChatInput'
import CitationSidebar from '../components/chat/CitationSidebar'

const SESSION_LANG_KEY = 'chiranjeevi_lang'

export default function ChatPage() {
    // ── Language state ─────────────────────────────────────────────
    const [language, setLanguage] = useState(() => sessionStorage.getItem(SESSION_LANG_KEY))
    const [showLangOverlay, setShowLangOverlay] = useState(() => !sessionStorage.getItem(SESSION_LANG_KEY))

    // ── Citation sidebar ────────────────────────────────────────────
    const [activeCitations, setActiveCitations] = useState(null)

    // ── Chat session hook ───────────────────────────────────────────
    const {
        messages, isProcessing, typingState,
        connectionStatus, sendMessage, checkConnection,
    } = useChatSession()

    // ── Speech recognition hook ─────────────────────────────────────
    const {
        isListening, isSupported, startListening, stopListening,
    } = useSpeechRecognition()

    // Check connection on mount
    useEffect(() => {
        checkConnection()
    }, [checkConnection])

    // ── Handlers ────────────────────────────────────────────────────
    const handleSelectLanguage = (code) => {
        setLanguage(code)
        setShowLangOverlay(false)
        sessionStorage.setItem(SESSION_LANG_KEY, code)
    }

    const handleChangeLang = () => {
        setShowLangOverlay(true)
    }

    const handleSend = (text) => {
        sendMessage(text, language || 'en-IN')
    }

    const handleMicClick = () => {
        if (isListening) {
            stopListening()
            return
        }
        startListening(language || 'en-IN', (transcript) => {
            handleSend(transcript)
        })
    }

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            position: 'relative',
            overflow: 'hidden',
        }}>
            {/* ── Chat panel ──────────────────────────────────────────── */}
            <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                background: 'var(--md-surface-container-lowest)',
                borderRadius: '24px',
                overflow: 'hidden',
                position: 'relative',
                boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
            }}>
                {/* Header */}
                <ChatHeader
                    connectionStatus={connectionStatus}
                    language={language || 'en-IN'}
                    onChangeLang={handleChangeLang}
                />

                {/* Messages */}
                <MessageList
                    messages={messages}
                    onCitationClick={setActiveCitations}
                />

                {/* Typing indicator (Forced ON for UI testing) */}
                <TypingIndicator
                    visible={true} 
                    label="🩺 Dr. Chiranjeevi is thinking"
                />

                {/* Input */}
                <ChatInput
                    onSend={handleSend}
                    isProcessing={isProcessing}
                    isListening={isListening}
                    onMicClick={handleMicClick}
                    isMicSupported={isSupported}
                />

                {/* Language overlay (absolute, covers entire chat panel) */}
                <LanguageOverlay
                    visible={showLangOverlay}
                    onSelect={handleSelectLanguage}
                />

                {/* Citation sidebar (absolute, inside chat panel) */}
                <CitationSidebar
                    citations={activeCitations}
                    onClose={() => setActiveCitations(null)}
                />
            </div>
        </div>
    )
}

/**
 * useSpeechRecognition.js
 * Wraps the browser Web Speech API for multilingual STT.
 */
import { useState, useRef, useCallback } from 'react'

export function useSpeechRecognition() {
    const [isListening, setIsListening] = useState(false)
    const [transcript, setTranscript] = useState('')
    const recognitionRef = useRef(null)

    const isSupported =
        typeof window !== 'undefined' &&
        !!(window.SpeechRecognition || window.webkitSpeechRecognition)

    const startListening = useCallback((lang = 'en-IN', onResult, onError) => {
        if (!isSupported) return
        const SpeechRecognition =
            window.SpeechRecognition || window.webkitSpeechRecognition
        const recognition = new SpeechRecognition()
        recognition.lang = lang
        recognition.continuous = false
        recognition.interimResults = false
        recognition.maxAlternatives = 1
        recognitionRef.current = recognition

        recognition.onstart = () => setIsListening(true)

        recognition.onresult = (event) => {
            const text = event.results[0][0].transcript
            setTranscript(text)
            onResult?.(text)
        }

        recognition.onend = () => setIsListening(false)

        recognition.onerror = (event) => {
            setIsListening(false)
            onError?.(event.error)
        }

        recognition.start()
    }, [isSupported])

    const stopListening = useCallback(() => {
        recognitionRef.current?.stop()
        setIsListening(false)
    }, [])

    return { isListening, transcript, isSupported, startListening, stopListening }
}

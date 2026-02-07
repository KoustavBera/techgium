import medicalChatService from '../services/MedicalChatService.js';
import Conversation from '../models/conversation.model.js';

/**
 * Socket.IO Chat Handler for Medical Chatbot
 * Supports two modes:
 * 1. Standalone: General medical Q&A
 * 2. Context-aware: Q&A with patient's medical report context
 */
export const handleChat = (socket, io) => {
    console.log(`[ChatHandler] New connection: ${socket.id}`);

    // Session state
    let sessionState = {
        sessionId: socket.id,
        mode: 'standalone',
        medicalContext: null,
        patientId: null,
        userId: null,
        language: 'en'
    };

    /**
     * Initialize chat session
     * Client sends: { userId, patientId, mode }
     */
    socket.on('init_session', async (data) => {
        try {
            const { userId, patientId, mode = 'standalone' } = data;

            sessionState.userId = userId;
            sessionState.patientId = patientId;
            sessionState.mode = mode;

            console.log(`[ChatHandler] Session initialized: ${JSON.stringify(sessionState)}`);

            // If context-aware mode and patientId provided, fetch medical report
            if (mode === 'context-aware' && patientId) {
                const reportContext = await medicalChatService.fetchPatientReport(patientId);

                if (reportContext) {
                    sessionState.medicalContext = reportContext;

                    socket.emit('session_initialized', {
                        success: true,
                        mode: 'context-aware',
                        message: 'Your medical report has been loaded. You can now ask questions about it.',
                        hasContext: true
                    });
                } else {
                    // Fallback to standalone if report not found
                    sessionState.mode = 'standalone';
                    socket.emit('session_initialized', {
                        success: true,
                        mode: 'standalone',
                        message: 'Medical report not found. Starting in general Q&A mode.',
                        hasContext: false
                    });
                }
            } else {
                socket.emit('session_initialized', {
                    success: true,
                    mode: 'standalone',
                    message: 'Welcome! Ask me any health-related questions.',
                    hasContext: false
                });
            }

            // Load or create conversation in database
            let conversation = await Conversation.findOne({ sessionId: socket.id });
            if (!conversation) {
                conversation = new Conversation({
                    sessionId: socket.id,
                    userId: userId || null,
                    patientId: patientId || null,
                    mode: sessionState.mode,
                    medicalContext: sessionState.medicalContext
                });
                await conversation.save();
            }

        } catch (error) {
            console.error('[ChatHandler] Session init error:', error);
            socket.emit('error', {
                message: 'Failed to initialize chat session',
                error: error.message
            });
        }
    });

    /**
     * Handle incoming chat messages
     * Client sends: { text, language }
     */
    socket.on('send_message', async (data) => {
        try {
            const { text, language } = data;

            if (!text || !text.trim()) {
                socket.emit('error', { message: 'Message cannot be empty' });
                return;
            }

            console.log(`[ChatHandler] Received message: "${text.substring(0, 50)}..."`);

            // Update language preference
            if (language) {
                sessionState.language = language;
            }

            // Show typing indicator
            socket.emit('typing', { isTyping: true });

            // Get response from MedicalChatService (3-pipeline flow)
            const result = await medicalChatService.chat(
                sessionState.sessionId,
                text,
                {
                    medicalContext: sessionState.medicalContext,
                    mode: sessionState.mode
                }
            );

            // Save to database
            const conversation = await Conversation.findOne({ sessionId: socket.id });
            if (conversation) {
                conversation.messages.push({
                    role: 'user',
                    content: text,
                    language: result.language,
                    timestamp: new Date()
                });
                conversation.messages.push({
                    role: 'assistant',
                    content: result.response,
                    language: result.language,
                    timestamp: new Date(),
                    isEmergency: result.isEmergency
                });
                await conversation.save();
            }

            // Hide typing indicator
            socket.emit('typing', { isTyping: false });

            // Send response to client
            socket.emit('receive_message', {
                text: result.response,
                sender: 'assistant',
                language: result.language,
                isEmergency: result.isEmergency,
                timestamp: new Date()
            });

            // If emergency detected, send alert
            if (result.isEmergency) {
                socket.emit('emergency_alert', {
                    message: '⚠️ This conversation indicates a potentially serious medical condition. Please consult a doctor immediately.',
                    severity: 'high'
                });

                // Optionally notify admin/doctor
                io.emit('admin_alert', {
                    sessionId: socket.id,
                    userId: sessionState.userId,
                    patientId: sessionState.patientId,
                    message: 'Emergency detected in chat session'
                });
            }

        } catch (error) {
            console.error('[ChatHandler] Message error:', error);
            socket.emit('typing', { isTyping: false });
            socket.emit('error', {
                message: 'Failed to process your message. Please try again.',
                error: error.message
            });
        }
    });

    /**
     * Change language preference
     * Client sends: { language: 'en' | 'hi' }
     */
    socket.on('change_language', (data) => {
        const { language } = data;
        sessionState.language = language;

        socket.emit('language_changed', {
            language,
            message: language === 'hi' ? 'भाषा बदल दी गई है' : 'Language changed'
        });
    });

    /**
     * Switch chat mode
     * Client sends: { mode: 'standalone' | 'context-aware', patientId }
     */
    socket.on('switch_mode', async (data) => {
        try {
            const { mode, patientId } = data;

            if (mode === 'context-aware' && patientId) {
                const reportContext = await medicalChatService.fetchPatientReport(patientId);

                if (reportContext) {
                    sessionState.mode = 'context-aware';
                    sessionState.medicalContext = reportContext;
                    sessionState.patientId = patientId;

                    socket.emit('mode_switched', {
                        success: true,
                        mode: 'context-aware',
                        message: 'Switched to context-aware mode. Your medical report is now loaded.'
                    });
                } else {
                    socket.emit('error', {
                        message: 'Medical report not found for this patient ID'
                    });
                }
            } else {
                sessionState.mode = 'standalone';
                sessionState.medicalContext = null;
                sessionState.patientId = null;

                socket.emit('mode_switched', {
                    success: true,
                    mode: 'standalone',
                    message: 'Switched to standalone mode. General medical Q&A.'
                });
            }

        } catch (error) {
            console.error('[ChatHandler] Mode switch error:', error);
            socket.emit('error', {
                message: 'Failed to switch mode',
                error: error.message
            });
        }
    });

    /**
     * Get conversation history
     */
    socket.on('get_history', async () => {
        try {
            const conversation = await Conversation.findOne({ sessionId: socket.id });

            if (conversation) {
                socket.emit('history_loaded', {
                    messages: conversation.messages,
                    metadata: conversation.metadata
                });
            } else {
                socket.emit('history_loaded', {
                    messages: [],
                    metadata: {}
                });
            }
        } catch (error) {
            console.error('[ChatHandler] History load error:', error);
            socket.emit('error', {
                message: 'Failed to load conversation history',
                error: error.message
            });
        }
    });

    /**
     * Clear conversation history
     */
    socket.on('clear_history', async () => {
        try {
            medicalChatService.clearHistory(socket.id);

            await Conversation.findOneAndUpdate(
                { sessionId: socket.id },
                { $set: { messages: [] } }
            );

            socket.emit('history_cleared', {
                success: true,
                message: 'Conversation history cleared'
            });
        } catch (error) {
            console.error('[ChatHandler] Clear history error:', error);
            socket.emit('error', {
                message: 'Failed to clear history',
                error: error.message
            });
        }
    });

    /**
     * Handle disconnection
     */
    socket.on('disconnect', async () => {
        console.log(`[ChatHandler] Client disconnected: ${socket.id}`);

        // Clear in-memory history
        medicalChatService.clearHistory(socket.id);

        // Update database
        try {
            await Conversation.findOneAndUpdate(
                { sessionId: socket.id },
                { $set: { 'metadata.lastActivity': new Date() } }
            );
        } catch (error) {
            console.error('[ChatHandler] Disconnect update error:', error);
        }
    });
};

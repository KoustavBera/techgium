import medicalChatService from '../services/MedicalChatService.js';

/**
 * Socket.IO Chat Handler - Uses LangChain for context
 */
export const handleChat = (socket, io) => {
    console.log(`[Chat] Connected: ${socket.id}`);

    // Session state
    let session = {
        id: socket.id,
        mode: 'standalone',
        medicalContext: null,
        patientId: null,
        userId: null
    };

    /**
     * Initialize session
     */
    socket.on('init_session', async (data = {}) => {
        try {
            const { userId, patientId, mode = 'standalone' } = data;

            session.userId = userId;
            session.patientId = patientId;
            session.mode = mode;

            // Fetch report for context-aware mode
            if (mode === 'context-aware' && patientId) {
                const report = await medicalChatService.fetchPatientReport(patientId);
                if (report) {
                    session.medicalContext = report;
                    socket.emit('session_initialized', {
                        success: true,
                        mode: 'context-aware',
                        message: 'Medical report loaded. Ask questions about it.',
                        hasContext: true,
                        session_id: socket.id
                    });
                } else {
                    session.mode = 'standalone';
                    socket.emit('session_initialized', {
                        success: true,
                        mode: 'standalone',
                        message: 'Report not found. General Q&A mode.',
                        hasContext: false,
                        session_id: socket.id
                    });
                }
            } else {
                socket.emit('session_initialized', {
                    success: true,
                    mode: 'standalone',
                    message: 'Welcome! Ask any health question.',
                    hasContext: false,
                    session_id: socket.id
                });
            }

            console.log(`[Chat] Session init: ${socket.id}, mode: ${session.mode}`);
        } catch (error) {
            console.error('[Chat] Init error:', error);
            socket.emit('error', { message: 'Session init failed' });
        }
    });

    /**
     * Handle messages
     */
    socket.on('send_message', async (data) => {
        try {
            const { text } = data;
            if (!text?.trim()) {
                socket.emit('error', { message: 'Empty message' });
                return;
            }

            console.log(`[Chat] Message: "${text.substring(0, 30)}..."`);
            socket.emit('typing', { isTyping: true });

            // Get response (uses LangChain memory internally)
            const result = await medicalChatService.chat(
                session.id,
                text,
                {
                    medicalContext: session.medicalContext,
                    mode: session.mode
                }
            );

            socket.emit('typing', { isTyping: false });
            socket.emit('receive_message', {
                text: result.response,
                sender: 'assistant',
                language: result.language,
                isEmergency: result.isEmergency,
                timestamp: new Date()
            });

            // Emergency alert
            if (result.isEmergency) {
                socket.emit('emergency_alert', {
                    message: '⚠️ Serious condition detected. Consult a doctor immediately.',
                    severity: 'high'
                });
                io.emit('admin_alert', {
                    sessionId: socket.id,
                    userId: session.userId,
                    message: 'Emergency in chat'
                });
            }

        } catch (error) {
            console.error('[Chat] Message error:', error);
            socket.emit('typing', { isTyping: false });
            socket.emit('error', { message: 'Failed to process message' });
        }
    });

    /**
     * Switch mode
     */
    socket.on('switch_mode', async (data) => {
        try {
            const { mode, patientId } = data;

            if (mode === 'context-aware' && patientId) {
                const report = await medicalChatService.fetchPatientReport(patientId);
                if (report) {
                    session.mode = 'context-aware';
                    session.medicalContext = report;
                    session.patientId = patientId;
                    socket.emit('mode_switched', { success: true, mode: 'context-aware' });
                } else {
                    socket.emit('error', { message: 'Report not found' });
                }
            } else {
                session.mode = 'standalone';
                session.medicalContext = null;
                socket.emit('mode_switched', { success: true, mode: 'standalone' });
            }
        } catch (error) {
            socket.emit('error', { message: 'Mode switch failed' });
        }
    });

    /**
     * Get history
     */
    socket.on('get_history', async () => {
        const history = await medicalChatService.getHistory(socket.id);
        const formatted = history.map(msg => ({
            role: msg.constructor.name.includes('Human') ? 'user' : 'assistant',
            content: msg.content
        }));
        socket.emit('history_loaded', { messages: formatted });
    });

    /**
     * Clear history
     */
    socket.on('clear_history', () => {
        medicalChatService.clearHistory(socket.id);
        socket.emit('history_cleared', { success: true });
    });

    /**
     * Disconnect
     */
    socket.on('disconnect', () => {
        console.log(`[Chat] Disconnected: ${socket.id}`);
        medicalChatService.clearHistory(socket.id);
    });
};

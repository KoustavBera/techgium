import { OpenAI } from "openai";
import dotenv from "dotenv";
import vectorStoreService from "./VectorStoreService.js";

dotenv.config();

/**
 * Medical Chatbot Service using LangChain-style pipeline
 * 
 * Architecture:
 * 1. Translation Pipeline: Hindi/Hinglish → English (Cohere)
 * 2. Medical Reasoning: English medical Q&A (gpt-oss-120b)
 * 3. Translation Pipeline: English → Original Language (Cohere)
 */
class MedicalChatService {
    constructor() {
        // Initialize OpenAI client for Hugging Face router
        this.client = new OpenAI({
            baseURL: "https://router.huggingface.co/v1",
            apiKey: process.env.HF_TOKEN,
        });

        // Model configurations
        this.models = {
            medical: "openai/gpt-oss-120b:groq",
            translate: "CohereLabs/command-a-translate-08-2025:cohere"
        };

        // Conversation history: sessionId -> messages[]
        this.conversationHistory = new Map();

        // Emergency keywords for alert detection
        this.emergencyKeywords = [
            'chest pain', 'heart attack', 'stroke', 'seizure', 'unconscious',
            'severe bleeding', 'difficulty breathing', 'suicide', 'overdose',
            'severe headache', 'paralysis', 'high fever', 'blood pressure'
        ];
    }

    /**
     * Detect language of input text
     * @param {string} text - Input text
     * @returns {string} - 'hi' for Hindi/Hinglish, 'en' for English
     */
    detectLanguage(text) {
        // Check for Devanagari script (Hindi)
        const hindiPattern = /[\u0900-\u097F]/;
        if (hindiPattern.test(text)) {
            return 'hi';
        }

        // Check for common Hindi words in Latin script (Hinglish)
        const hinglishWords = ['kya', 'hai', 'mujhe', 'aap', 'hum', 'main', 'ke', 'ki', 'ko', 'se'];
        const lowerText = text.toLowerCase();
        const hasHinglish = hinglishWords.some(word => lowerText.includes(word));

        if (hasHinglish) {
            return 'hi';
        }

        return 'en';
    }

    /**
     * Pipeline 1 & 3: Translation using Cohere
     * @param {string} text - Text to translate
     * @param {string} sourceLang - Source language ('hi' or 'en')
     * @param {string} targetLang - Target language ('hi' or 'en')
     * @returns {Promise<string>} - Translated text
     */
    async translate(text, sourceLang, targetLang) {
        // Skip translation if same language
        if (sourceLang === targetLang) {
            return text;
        }

        try {
            const translationPrompt = sourceLang === 'hi'
                ? `Translate the following Hindi/Hinglish text to English. Only provide the translation, no explanations:\n\n${text}`
                : `Translate the following English text to Hindi. Only provide the translation, no explanations:\n\n${text}`;

            const response = await this.client.chat.completions.create({
                model: this.models.translate,
                messages: [
                    {
                        role: "system",
                        content: "You are a professional translator specializing in Hindi-English medical translations. Provide only the translation without any additional text."
                    },
                    {
                        role: "user",
                        content: translationPrompt,
                    },
                ],
                temperature: 0.3, // Low temperature for consistent translations
                max_tokens: 500,
            });

            return response.choices[0].message.content.trim();
        } catch (error) {
            console.error('Translation error:', error);
            return text; // Fallback to original text
        }
    }

    /**
     * Pipeline 2: Medical reasoning using gpt-oss-120b
     * @param {string} englishQuery - Medical query in English
     * @param {string} medicalContext - Patient's medical report context (optional)
     * @param {Array} conversationHistory - Previous conversation messages
     * @returns {Promise<Object>} - { response: string, isEmergency: boolean }
     */
    async getMedicalResponse(englishQuery, medicalContext = null, conversationHistory = []) {
        try {
            // Build system prompt
            let systemPrompt = `You are a professional medical AI assistant. Your role is to:
1. Answer medical and health-related questions accurately
2. Provide helpful health advice based on symptoms
3. Identify potentially serious conditions that require immediate medical attention
4. Always remind users that you are not a replacement for professional medical diagnosis

Important guidelines:
- Be empathetic and professional
- Use simple, clear language
- If symptoms suggest a serious condition, clearly state: "EMERGENCY: Please consult a doctor immediately"
- Never diagnose definitively, only provide information and suggestions
- Encourage users to seek professional medical help when appropriate`;

            // Add medical context if available
            if (medicalContext) {
                systemPrompt += `\n\nPatient's Medical Report Context:\n${medicalContext}`;
            }

            // Build messages array with conversation history
            const messages = [
                { role: "system", content: systemPrompt },
                ...conversationHistory,
                { role: "user", content: englishQuery }
            ];

            const response = await this.client.chat.completions.create({
                model: this.models.medical,
                messages: messages,
                temperature: 0.7,
                max_tokens: 800,
            });

            const medicalResponse = response.choices[0].message.content.trim();

            // Check for emergency keywords
            const isEmergency = this.detectEmergency(englishQuery, medicalResponse);

            return {
                response: medicalResponse,
                isEmergency
            };
        } catch (error) {
            console.error('Medical reasoning error:', error);
            throw new Error('Failed to get medical response. Please try again.');
        }
    }

    /**
     * Detect if query or response indicates an emergency
     * @param {string} query - User's query
     * @param {string} response - AI's response
     * @returns {boolean} - True if emergency detected
     */
    detectEmergency(query, response) {
        const combinedText = (query + ' ' + response).toLowerCase();

        // Check for emergency keywords
        const hasEmergencyKeyword = this.emergencyKeywords.some(keyword =>
            combinedText.includes(keyword.toLowerCase())
        );

        // Check for explicit emergency markers in response
        const hasEmergencyMarker = response.includes('EMERGENCY:') ||
            response.includes('immediately') ||
            response.includes('urgent medical attention');

        return hasEmergencyKeyword || hasEmergencyMarker;
    }

    /**
     * Main chat function - orchestrates the 3-pipeline flow with RAG
     * @param {string} sessionId - Unique session identifier
     * @param {string} userMessage - User's input message
     * @param {Object} options - { medicalContext, mode, userId }
     * @returns {Promise<Object>} - { response, language, isEmergency }
     */
    async chat(sessionId, userMessage, options = {}) {
        const { medicalContext = null, mode = 'standalone', userId = null } = options;

        // Initialize conversation history for this session
        if (!this.conversationHistory.has(sessionId)) {
            this.conversationHistory.set(sessionId, []);
        }

        const history = this.conversationHistory.get(sessionId);

        // Step 1: Detect input language
        const inputLanguage = this.detectLanguage(userMessage);
        console.log(`[MedicalChat] Detected language: ${inputLanguage}`);

        // Step 2: Pipeline 1 - Translate to English if needed
        const englishQuery = await this.translate(userMessage, inputLanguage, 'en');
        console.log(`[MedicalChat] English query: ${englishQuery}`);

        // Step 3: RAG - Retrieve relevant context from vector store
        let ragContext = "";
        if (mode === 'standalone') {
            try {
                ragContext = await vectorStoreService.buildRAGContext(englishQuery, userId);
                if (ragContext) {
                    console.log(`[MedicalChat] RAG context retrieved: ${ragContext.substring(0, 100)}...`);
                }
            } catch (error) {
                console.error('[MedicalChat] RAG context error:', error);
            }
        }

        // Combine RAG context with any provided medical context
        const combinedContext = [ragContext, medicalContext].filter(Boolean).join('\n\n');

        // Step 4: Pipeline 2 - Get medical response with RAG context
        const { response: englishResponse, isEmergency } = await this.getMedicalResponse(
            englishQuery,
            combinedContext || null,
            history
        );
        console.log(`[MedicalChat] Medical response: ${englishResponse.substring(0, 100)}...`);

        // Step 5: Pipeline 3 - Translate back to original language
        const finalResponse = await this.translate(englishResponse, 'en', inputLanguage);
        console.log(`[MedicalChat] Final response: ${finalResponse.substring(0, 100)}...`);

        // Step 6: Store messages with embeddings for future RAG
        try {
            await vectorStoreService.storeMessageWithEmbedding(
                sessionId,
                userMessage,
                englishQuery,
                'user',
                { language: inputLanguage }
            );
            await vectorStoreService.storeMessageWithEmbedding(
                sessionId,
                finalResponse,
                englishResponse,
                'assistant',
                { language: inputLanguage, isEmergency }
            );
        } catch (error) {
            console.error('[MedicalChat] Store embedding error:', error);
        }

        // Update in-memory conversation history
        history.push({ role: "user", content: englishQuery });
        history.push({ role: "assistant", content: englishResponse });

        // Keep only last 10 messages to manage token limits
        if (history.length > 10) {
            history.splice(0, history.length - 10);
        }

        return {
            response: finalResponse,
            language: inputLanguage,
            isEmergency,
            mode,
            ragContextUsed: !!ragContext
        };
    }

    /**
     * Fetch patient's medical report from FastAPI service
     * @param {string} patientId - Patient ID
     * @returns {Promise<string>} - Medical report text
     */
    async fetchPatientReport(patientId) {
        try {
            // TODO: Replace with actual FastAPI endpoint
            const fastApiUrl = process.env.FASTAPI_URL || 'http://localhost:8000';
            const response = await fetch(`${fastApiUrl}/api/v1/reports/${patientId}`);

            if (!response.ok) {
                throw new Error('Failed to fetch patient report');
            }

            const reportData = await response.json();

            // Extract relevant medical information
            const reportSummary = this.extractReportSummary(reportData);
            return reportSummary;
        } catch (error) {
            console.error('Error fetching patient report:', error);
            return null;
        }
    }

    /**
     * Extract summary from patient report data
     * @param {Object} reportData - Full report data from FastAPI
     * @returns {string} - Summarized medical context
     */
    extractReportSummary(reportData) {
        // Extract key biomarkers and findings
        let summary = "Patient Medical Report Summary:\n\n";

        if (reportData.systems) {
            reportData.systems.forEach(system => {
                summary += `${system.system.toUpperCase()}:\n`;
                if (system.biomarkers) {
                    system.biomarkers.forEach(biomarker => {
                        summary += `- ${biomarker.name}: ${biomarker.value} ${biomarker.unit}\n`;
                    });
                }
                summary += '\n';
            });
        }

        if (reportData.risk_assessment) {
            summary += `Overall Risk Level: ${reportData.risk_assessment.overall_risk}\n`;
        }

        return summary;
    }

    /**
     * Clear conversation history for a session
     * @param {string} sessionId - Session ID to clear
     */
    clearHistory(sessionId) {
        this.conversationHistory.delete(sessionId);
    }

    /**
     * Get conversation history for a session
     * @param {string} sessionId - Session ID
     * @returns {Array} - Conversation history
     */
    getHistory(sessionId) {
        return this.conversationHistory.get(sessionId) || [];
    }
}

// Export singleton instance
export default new MedicalChatService();

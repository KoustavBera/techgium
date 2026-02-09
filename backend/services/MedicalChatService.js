import { OpenAI } from "openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import dotenv from "dotenv";

dotenv.config();

/**
 * Medical Chatbot Service using LangChain for Context Management
 * 
 * Architecture:
 * 1. Translation Pipeline: Hindi/Hinglish → English (Cohere)
 * 2. Medical Reasoning: English medical Q&A with LangChain memory (gpt-oss-120b)
 * 3. Translation Pipeline: English → Original Language (Cohere)
 */
class MedicalChatService {
    constructor() {
        // OpenAI client for Hugging Face router
        this.client = new OpenAI({
            baseURL: "https://router.huggingface.co/v1",
            apiKey: process.env.HF_TOKEN,
        });

        // Models
        this.models = {
            medical: "openai/gpt-oss-120b:groq",
            translate: "CohereLabs/command-a-translate-08-2025:cohere"
        };

        // LangChain memory map: sessionId -> InMemoryChatMessageHistory
        this.memoryStore = new Map();

        // Critical emergency keywords (always trigger emergency)
        this.criticalEmergencyKeywords = [
            'heart attack', 'stroke', 'seizure', 'unconscious', 'not breathing',
            'severe bleeding', 'suicide', 'overdose', 'paralysis', 'anaphylaxis'
        ];

        // Warning keywords (require LLM confirmation via EMERGENCY: marker)
        this.warningKeywords = [
            'chest pain', 'difficulty breathing', 'severe headache', 'high fever',
            'blood pressure', 'fainting', 'severe pain'
        ];

        // System prompt for medical assistant
        this.systemPrompt = `You are Dr. MedAssist, a friendly medical AI assistant for Indian users.

ROLE:
- Answer medical and health questions accurately
- Provide helpful advice, home remedies, and medication suggestions
- Identify TRULY serious conditions requiring immediate attention
- You are NOT a replacement for professional diagnosis

GUIDELINES:
- Be empathetic, warm, and conversational
- Use simple, easy-to-understand language
- For common ailments: suggest home remedies, OTC medicines, when to see a doctor
- ONLY use "EMERGENCY:" prefix for life-threatening conditions (heart attack, stroke, severe allergic reaction, etc.)
- High fever alone is NOT an emergency - suggest fever management + when to see a doctor
- Sore throat + fever = common cold/flu, suggest remedies first
- Never diagnose definitively
- Consider Indian medical context (AYUSH, Ayurvedic remedies, local medicines)`;
    }

    /**
     * Get or create LangChain memory for session
     */
    getMemory(sessionId) {
        if (!this.memoryStore.has(sessionId)) {
            this.memoryStore.set(sessionId, new InMemoryChatMessageHistory());
        }
        return this.memoryStore.get(sessionId);
    }

    /**
     * Detect language and style of input text
     * Returns: { language: 'en'|'hi', style: 'english'|'hindi'|'hinglish' }
     */
    detectLanguage(text) {
        // Hindi (Devanagari script)
        const hindiPattern = /[\u0900-\u097F]/;
        if (hindiPattern.test(text)) {
            return { language: 'hi', style: 'hindi' };
        }

        // Hinglish (common Hindi words in Latin script)
        const hinglishWords = [
            'kya', 'hai', 'mujhe', 'aap', 'hum', 'main', 'ke', 'ki', 'ko', 'se',
            'mera', 'tera', 'kaise', 'kahan', 'nahi', 'haan', 'acha', 'theek',
            'bahut', 'bhi', 'aur', 'lekin', 'toh', 'mein', 'ho', 'raha', 'rahi',
            'karo', 'karna', 'dard', 'bukhar', 'sir', 'pet', 'gala', 'khana',
            'paani', 'dawai', 'doctor', 'hospital', 'lagra', 'lagta', 'hora', 'hota'
        ];
        const lowerText = text.toLowerCase();
        const hinglishMatches = hinglishWords.filter(word => {
            const regex = new RegExp(`\\b${word}\\b`, 'i');
            return regex.test(lowerText);
        });

        if (hinglishMatches.length >= 1) {
            return { language: 'hi', style: 'hinglish' };
        }

        return { language: 'en', style: 'english' };
    }

    /**
     * Pipeline 1 & 3: Translation using Cohere
     * Preserves Hinglish style when input is Hinglish
     */
    async translate(text, sourceLang, targetLang, inputStyle = 'english') {
        if (sourceLang === targetLang) return text;

        try {
            let prompt, systemContent;

            if (sourceLang === 'hi' || inputStyle === 'hinglish') {
                // Translating Hindi/Hinglish TO English
                prompt = `Translate this Hindi/Hinglish text to clear English. Only provide the translation:\n\n${text}`;
                systemContent = "You are a Hindi-English medical translator. Provide only the translation, nothing else.";
            } else {
                // Translating English back - match the input style
                if (inputStyle === 'hinglish') {
                    prompt = `Convert this English medical response to Hinglish (Hindi words written in English/Roman script, casual conversational style). Mix Hindi and English naturally like how Indians speak. Only provide the translation:\n\n${text}`;
                    systemContent = "You are a medical translator. Convert to casual Hinglish (Roman script). Use common Hindi words mixed with English. Example: 'Aapko paani bahut peena chahiye aur rest lena hai. Fever ke liye Paracetamol le sakte ho.'";
                } else if (inputStyle === 'hindi') {
                    prompt = `Translate this English medical response to Hindi (Devanagari script). Only provide the translation:\n\n${text}`;
                    systemContent = "You are a Hindi-English medical translator. Translate to Hindi in Devanagari script. Provide only the translation.";
                } else {
                    return text; // English in, English out
                }
            }

            const response = await this.client.chat.completions.create({
                model: this.models.translate,
                messages: [
                    { role: "system", content: systemContent },
                    { role: "user", content: prompt }
                ],
                temperature: 0.4,
                max_tokens: 600,
            });

            return response.choices[0].message.content.trim();
        } catch (error) {
            console.error('[Translate] Error:', error.message);
            return text;
        }
    }

    /**
     * Pipeline 2: Medical reasoning with LangChain memory
     */
    async getMedicalResponse(sessionId, englishQuery, medicalContext = null) {
        try {
            const chatHistory = this.getMemory(sessionId);

            // Get chat history from LangChain memory
            const history = await chatHistory.getMessages();

            // Build messages array
            let systemContent = this.systemPrompt;
            if (medicalContext) {
                systemContent += `\n\nPatient Medical Context:\n${medicalContext}`;
            }

            const messages = [
                { role: "system", content: systemContent }
            ];

            // Add history from LangChain memory
            for (const msg of history) {
                if (msg instanceof HumanMessage) {
                    messages.push({ role: "user", content: msg.content });
                } else if (msg instanceof AIMessage) {
                    messages.push({ role: "assistant", content: msg.content });
                }
            }

            // Add current query
            messages.push({ role: "user", content: englishQuery });

            // Call LLM
            const response = await this.client.chat.completions.create({
                model: this.models.medical,
                messages: messages,
                temperature: 0.7,
                max_tokens: 800,
            });

            const medicalResponse = response.choices[0].message.content.trim();

            // Save to LangChain memory
            await chatHistory.addMessage(new HumanMessage(englishQuery));
            await chatHistory.addMessage(new AIMessage(medicalResponse));

            // Check emergency
            const isEmergency = this.detectEmergency(englishQuery, medicalResponse);

            return { response: medicalResponse, isEmergency };
        } catch (error) {
            console.error('[Medical] Error:', error.message);
            throw new Error('Failed to get medical response');
        }
    }

    /**
     * Detect emergency conditions - smarter detection
     */
    detectEmergency(query, response) {
        const combined = (query + ' ' + response).toLowerCase();

        // Check for critical keywords (always emergency)
        const hasCritical = this.criticalEmergencyKeywords.some(kw => combined.includes(kw));
        if (hasCritical) return true;

        // Check if LLM explicitly marked as emergency
        const hasLLMMarker = response.includes('EMERGENCY:') ||
            response.includes('call emergency') ||
            response.includes('call 108') ||
            response.includes('rush to hospital');

        return hasLLMMarker;
    }

    /**
     * Main chat function - 3-pipeline flow with LangChain context
     */
    async chat(sessionId, userMessage, options = {}) {
        const { medicalContext = null, mode = 'standalone' } = options;

        // Step 1: Detect language and style (english/hindi/hinglish)
        const { language: inputLanguage, style: inputStyle } = this.detectLanguage(userMessage);
        console.log(`[Chat] Language: ${inputLanguage}, Style: ${inputStyle}`);

        // Step 2: Translate to English (for LLM processing)
        const englishQuery = await this.translate(userMessage, inputLanguage, 'en', inputStyle);
        console.log(`[Chat] English: ${englishQuery.substring(0, 50)}...`);

        // Step 3: Get medical response with LangChain memory
        const { response: englishResponse, isEmergency } = await this.getMedicalResponse(
            sessionId,
            englishQuery,
            medicalContext
        );
        console.log(`[Chat] Response: ${englishResponse.substring(0, 50)}...`);

        // Step 4: Translate back to original style (preserves Hinglish if input was Hinglish)
        const finalResponse = await this.translate(englishResponse, 'en', inputLanguage, inputStyle);

        return {
            response: finalResponse,
            language: inputLanguage,
            style: inputStyle,
            isEmergency,
            mode
        };
    }

    /**
     * Fetch patient report from FastAPI
     */
    async fetchPatientReport(patientId) {
        try {
            const url = process.env.FASTAPI_URL || 'http://localhost:8000';
            const res = await fetch(`${url}/api/v1/reports/${patientId}`);
            if (!res.ok) throw new Error('Report not found');

            const data = await res.json();
            return this.formatReport(data);
        } catch (error) {
            console.error('[Report] Error:', error.message);
            return null;
        }
    }

    /**
     * Format patient report
     */
    formatReport(data) {
        let summary = "Patient Report:\n";
        if (data.systems) {
            data.systems.forEach(sys => {
                summary += `\n${sys.system.toUpperCase()}:\n`;
                sys.biomarkers?.forEach(b => {
                    summary += `- ${b.name}: ${b.value} ${b.unit}\n`;
                });
            });
        }
        if (data.risk_assessment) {
            summary += `\nRisk Level: ${data.risk_assessment.overall_risk}`;
        }
        return summary;
    }

    /**
     * Clear session memory
     */
    clearHistory(sessionId) {
        this.memoryStore.delete(sessionId);
    }

    /**
     * Get session history
     */
    async getHistory(sessionId) {
        const chatHistory = this.memoryStore.get(sessionId);
        if (!chatHistory) return [];
        return await chatHistory.getMessages();
    }
}

export default new MedicalChatService();

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

        // Enhanced System Prompt with Few-Shot Examples
        // Updated System Prompt to be more robust
        this.systemPrompt = `You are Dr. MedAssist, a compassionate and knowledgeable medical AI assistant.

## YOUR EXPERTISE
- General medicine & common ailments
- Indian medical context (Ayurveda, home remedies, OTC medicines)

## RESPONSE FORMAT
1. **Assessment**: What you think is happening
2. **Recommendations**: Actionable advice (bullet points)
3. **Warning Signs**: When to seek help

## IMPORTANT RULES
- Be warm and empathetic
- Suggest home remedies & Indian OTC medicines (Crocin, Digene, etc.)
- ONLY say "EMERGENCY:" for life-threatening situations
- Keep responses concise and easy to read`;
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

        // Expanded Hinglish words list (common verbs, nouns, pronouns)
        const hinglishWords = [
            'kya', 'hai', 'mujhe', 'aap', 'hum', 'main', 'ke', 'ki', 'ko', 'se',
            'mera', 'tera', 'kaise', 'kahan', 'nahi', 'haan', 'acha', 'theek',
            'bahut', 'bhi', 'aur', 'lekin', 'toh', 'mein', 'ho', 'raha', 'rahi',
            'karo', 'karna', 'dard', 'bukhar', 'sir', 'pet', 'gala', 'khana',
            'paani', 'dawai', 'lagra', 'lagta', 'hora', 'hota',
            'batao', 'bataiye', 'suno', 'dekho', 'kuch', 'kadwa', 'aaj', 'kal',
            'din', 'raat', 'subah', 'shaam', 'dopahar', 'kab', 'kyu', 'kaun',
            'tablet', 'goli', 'ilaj', 'upay'
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
                // Added instruction to preserve medical terms if needed, but primary goal is clear English for the LLM
                prompt = `Translate the following Hindi or Hinglish text into clear, concise English for a medical AI.
If the user mentions specific medical terms (like 'sar dard', 'bukhar'), ensure they are translated correctly (headache, fever).
Only provide the English translation, do not add any conversational filler.

Text to translate:
"${text}"`;
                systemContent = "You are an expert medical translator. Translate Hindi/Hinglish to English accurately. Direct translation only.";
            } else {
                // Translating English back - match the input style
                if (inputStyle === 'hinglish') {
                    // STRICT Hinglish prompt
                    prompt = `Translate the following medical response into conversational Hinglish (Hindi written in English alphabet).
Use common Indian English terms where appropriate, but the grammar and core vocabulary should be Hindi.
Example: "You should drink water" -> "Aapko paani peena chahiye".
Make it sound like a friendly Indian doctor speaking to a patient.

Medical Response to translate:
"${text}"`;
                    systemContent = "You are a friendly Indian medical assistant. You speak in Hinglish (Roman Hindi). NEVER switch to pure English sentences. Always use Hindi grammar with English medical terms if needed (e.g., 'Blood pressure check karwana chahiye').";
                } else if (inputStyle === 'hindi') {
                    prompt = `Translate this English medical response to Hindi (Devanagari script). Maintain a polite and professional doctor's tone.

Medical Response:
"${text}"`;
                    systemContent = "You are a professional medical translator. Translate to formal Hindi (Devanagari).";
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

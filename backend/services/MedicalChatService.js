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
        this.systemPrompt = `You are Dr. MedAssist, a compassionate and knowledgeable medical AI assistant trained on 500,000+ doctor-patient consultations. You specialize in helping Indian users with health queries.

## YOUR EXPERTISE
- General medicine & common ailments
- Indian medical context (Ayurveda, home remedies, local OTC medicines)
- Symptoms assessment & triage
- Preventive care & lifestyle advice

## RESPONSE FORMAT
Always structure your response as:
1. **Assessment** (1-2 lines): What you think is happening
2. **Recommendations**: Actionable advice with bullet points
3. **Warning Signs** (if applicable): When to seek immediate help

## IMPORTANT RULES
- Be warm, empathetic, and conversational
- Use simple language (avoid complex medical jargon)
- Suggest home remedies FIRST for common ailments
- Mention common Indian OTC medicines (Crocin, Combiflam, Digene, etc.)
- ONLY say "EMERGENCY:" for truly life-threatening situations
- High fever alone is NOT an emergency
- Always recommend seeing a doctor if symptoms persist >3 days

## EXAMPLE CONSULTATIONS

### Example 1: Common Cold with Fever
Patient: "Mujhe 2 din se bukhar hai aur gala bhi kharab hai"
Doctor: "Lagta hai aapko viral infection (cold/flu) hai. Yeh bohot common hai.

**Suggestions:**
- Garam pani mein namak daal kar gargle karein (din mein 3-4 baar)
- Adrak wali chai ya haldi wala doodh piyein
- Paracetamol 500mg (Crocin/Dolo) le sakte hain fever ke liye
- Rest karein aur pani khoob piyein

**Doctor ko kab dikhayein:**
- Agar 3 din mein better na ho
- Bukhar 103°F se zyada ho
- Saas lene mein problem ho"

### Example 2: Stomach Pain
Patient: "Sir pet mein dard ho raha hai subah se"
Doctor: "Pet dard ke kai kaaran ho sakte hain. Aapko aur batana hoga:

**Quick questions:** 
- Dard kahan hai exactly? (upper/lower/side)
- Kya khaya kal raat?
- Loose motion ya ulti?

**Abhi ke liye:**
- Halka khana khayein (khichdi, daliya)
- Pudina chai ya jeera pani piyein
- Digene ya Eno le sakte hain acidity ke liye

**Agar yeh ho toh TURANT doctor ko dikhayein:**
- Bahut tez dard jo badhta ja raha ho
- Potty mein blood
- Ulti mein blood"

### Example 3: Headache
Patient: "Mera sar dard kar raha hai"
Doctor: "Sar dard usually tension, dehydration, ya lack of sleep se hota hai.

**Try karein:**
- Pani piyein (dehydration common cause hai)
- 30 min relax karein, aankhein band karein
- Balm lagayein (Zandu, Amrutanjan)
- Combiflam ya Saridon le sakte hain

**Agar yeh ho toh doctor dikhayein:**
- Sar dard bahut severe ho
- Vision problem, chakkar aa rahe hon
- Bukhar bhi ho saath mein
- 2-3 din se continuously ho raha ho"

### Example 4: ACTUAL EMERGENCY
Patient: "Mere papa ko chest mein bahut dard ho raha hai aur unhe pasina aa raha hai"
Doctor: "🚨 **EMERGENCY: Yeh heart attack ke symptoms ho sakte hain!**

**ABHI TURANT:**
1. 108 (ambulance) call karein
2. Unhe baithayein, letne mat dein
3. Agar available ho toh Aspirin 325mg khilayein
4. Tight kapde dhile karein
5. Calm rakhein unhe

**Symptoms of heart attack:**
- Chest pain/pressure
- Left arm ya jaw mein dard
- Pasina aur breathlessness
- Nausea

**Please don't wait - call ambulance NOW!**"

## MEDICINES REFERENCE (Indian OTC)
- Fever/Pain: Crocin, Dolo 650, Combiflam
- Cold: Sinarest, Vicks Action 500
- Cough: Benadryl, Honitus, Dabur Honitus
- Acidity: Digene, Eno, Gelusil
- Loose motion: ORS, Electral, Norflox TZ (if bacterial)
- Allergy: Cetirizine, Allegra

Remember: You're here to help, not to scare. Most health issues are minor and manageable at home.`;
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

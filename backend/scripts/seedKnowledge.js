import vectorStoreService from '../services/VectorStoreService.js';

/**
 * Seed the medical knowledge base with initial data
 * Run this script once to populate the knowledge base
 */
async function seedMedicalKnowledge() {
    console.log('[Seed] Starting medical knowledge base seeding...');

    const medicalKnowledge = [
        // Symptoms
        {
            category: 'symptom',
            content: 'Headache can be caused by tension, dehydration, lack of sleep, eye strain, or more serious conditions like migraines or high blood pressure. Common remedies include rest, hydration, over-the-counter pain relievers, and avoiding bright screens.',
            summary: 'Headache causes include tension, dehydration, lack of sleep. Treatment: rest, hydration, OTC pain relievers.',
            keywords: ['headache', 'sar dard', 'migraine', 'tension', 'pain']
        },
        {
            category: 'symptom',
            content: 'Fever indicates the body is fighting an infection. Normal body temperature is 98.6°F (37°C). Fever above 103°F (39.4°C) requires immediate medical attention. Treatment includes rest, fluids, and antipyretics like paracetamol or ibuprofen.',
            summary: 'Fever indicates infection. High fever (>103°F) needs immediate attention. Treatment: rest, fluids, antipyretics.',
            keywords: ['fever', 'bukhar', 'temperature', 'high fever', 'infection']
        },
        {
            category: 'symptom',
            content: 'Chest pain can indicate heart attack, angina, acid reflux, or muscle strain. Heart attack symptoms include pain radiating to arm, shortness of breath, sweating. Seek emergency care immediately for suspected heart issues.',
            summary: 'Chest pain may indicate heart attack, angina, or GERD. Emergency symptoms: arm pain, breathlessness, sweating.',
            keywords: ['chest pain', 'seene mein dard', 'heart', 'pain', 'breathing']
        },
        {
            category: 'symptom',
            content: 'Cough can be dry or productive. Common causes are viral infections, allergies, asthma, or GERD. Persistent cough lasting more than 3 weeks requires medical evaluation. Home remedies include honey, warm fluids, and steam inhalation.',
            summary: 'Cough types: dry or productive. Causes: infections, allergies, asthma. Persistent cough (>3 weeks) needs evaluation.',
            keywords: ['cough', 'khansi', 'dry cough', 'cold', 'throat']
        },

        // Conditions
        {
            category: 'condition',
            content: 'Diabetes is a chronic condition affecting blood sugar regulation. Type 1 is autoimmune; Type 2 is lifestyle-related. Symptoms include frequent urination, thirst, fatigue. Management involves diet, exercise, and medication. Regular monitoring is essential.',
            summary: 'Diabetes affects blood sugar. Type 1 (autoimmune) vs Type 2 (lifestyle). Symptoms: thirst, fatigue, urination.',
            keywords: ['diabetes', 'sugar', 'madhumeh', 'blood sugar', 'insulin']
        },
        {
            category: 'condition',
            content: 'Hypertension (high blood pressure) is a silent condition with no early symptoms. Normal BP is 120/80 mmHg. High BP above 140/90 mmHg increases risk of heart disease and stroke. Management includes low-salt diet, exercise, and medication.',
            summary: 'Hypertension is silent. Normal BP: 120/80. High: >140/90. Risks: heart disease, stroke. Manage with diet/exercise.',
            keywords: ['blood pressure', 'BP', 'hypertension', 'high BP', 'low BP']
        },

        // Treatments
        {
            category: 'treatment',
            content: 'Paracetamol (Acetaminophen) is used for fever and mild pain relief. Adult dose is 500-1000mg every 4-6 hours, max 4g/day. Contraindicated in liver disease. Side effects include nausea and allergic reactions in rare cases.',
            summary: 'Paracetamol for fever/pain. Dose: 500-1000mg every 4-6 hrs, max 4g/day. Avoid in liver disease.',
            keywords: ['paracetamol', 'crocin', 'dolo', 'fever medicine', 'pain relief']
        },
        {
            category: 'treatment',
            content: 'Oral Rehydration Solution (ORS) is used for dehydration from diarrhea or vomiting. Mix 1 packet in 1 liter of clean water. Drink small sips frequently. Seek medical care if symptoms persist beyond 24 hours or if blood is present.',
            summary: 'ORS for dehydration. Mix 1 packet in 1L water, drink in small sips. Seek care if symptoms persist 24+ hours.',
            keywords: ['ORS', 'dehydration', 'diarrhea', 'dast', 'vomiting', 'ulti']
        },

        // Emergency
        {
            category: 'emergency',
            content: 'Signs of a heart attack: severe chest pain or pressure, pain radiating to left arm/jaw, shortness of breath, sweating, nausea. CALL EMERGENCY IMMEDIATELY. Have patient chew aspirin if available. Do CPR if unresponsive.',
            summary: 'Heart attack signs: chest pain, arm/jaw pain, breathlessness. CALL EMERGENCY. Give aspirin. CPR if unresponsive.',
            keywords: ['heart attack', 'emergency', 'chest pain', 'CPR', 'ambulance']
        },
        {
            category: 'emergency',
            content: 'Stroke symptoms: FAST - Face drooping, Arm weakness, Speech difficulty, Time to call emergency. Other signs include sudden confusion, vision problems, severe headache. Time is critical - treatment within 3 hours improves outcomes.',
            summary: 'Stroke: FAST (Face, Arm, Speech, Time). Signs: confusion, vision issues, headache. Call emergency within 3 hours.',
            keywords: ['stroke', 'paralysis', 'lakwa', 'face drooping', 'emergency']
        },

        // General
        {
            category: 'general',
            content: 'Healthy lifestyle includes: 7-8 hours sleep, balanced diet with fruits and vegetables, 30 minutes daily exercise, adequate water intake (8 glasses), stress management, and regular health checkups. Avoid smoking and excessive alcohol.',
            summary: 'Healthy lifestyle: 7-8 hrs sleep, balanced diet, 30 min exercise, 8 glasses water, manage stress.',
            keywords: ['health tips', 'lifestyle', 'exercise', 'diet', 'sleep']
        },
        {
            category: 'general',
            content: 'Common cold usually lasts 7-10 days. Symptoms include runny nose, sneezing, sore throat. Rest, fluids, and OTC decongestants help. Antibiotics are not effective against viral infections. See a doctor if fever exceeds 103°F or symptoms worsen.',
            summary: 'Common cold lasts 7-10 days. Treatment: rest, fluids, decongestants. Antibiotics ineffective. See doctor if fever >103°F.',
            keywords: ['cold', 'sardi', 'runny nose', 'sneezing', 'sore throat']
        }
    ];

    let successCount = 0;
    let errorCount = 0;

    for (const knowledge of medicalKnowledge) {
        try {
            await vectorStoreService.addMedicalKnowledge(knowledge);
            successCount++;
            console.log(`[Seed] Added: ${knowledge.summary.substring(0, 50)}...`);
        } catch (error) {
            errorCount++;
            console.error(`[Seed] Error adding: ${knowledge.summary.substring(0, 50)}...`, error.message);
        }
    }

    console.log(`[Seed] Completed. Success: ${successCount}, Errors: ${errorCount}`);
}

// Export for use
export { seedMedicalKnowledge };

// Run if executed directly
if (process.argv[1].includes('seedKnowledge.js')) {
    import('../config/connectDB.js').then(async (module) => {
        await module.default();
        await seedMedicalKnowledge();
        process.exit(0);
    }).catch(console.error);
}

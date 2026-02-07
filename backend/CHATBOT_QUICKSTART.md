# Medical Chatbot Integration - Quick Start Guide

## 🚀 What's Been Implemented

### Backend Services
1. **MedicalChatService.js** - Core 3-pipeline architecture:
   - Pipeline 1: Hindi/Hinglish → English translation (Cohere)
   - Pipeline 2: Medical reasoning (gpt-oss-120b)
   - Pipeline 3: English → Original language translation (Cohere)

2. **chatHandler.js** - Socket.IO event handlers:
   - `init_session` - Initialize chat with optional patient context
   - `send_message` - Process user messages through 3-pipeline
   - `change_language` - Switch between Hindi/English
   - `switch_mode` - Toggle standalone/context-aware modes
   - `get_history` - Retrieve conversation history
   - `clear_history` - Clear conversation
   - Emergency detection & alerts

3. **conversation.model.js** - MongoDB schema for storing conversations

---

## 📋 Setup Instructions

### 1. Install Dependencies
```bash
cd backend
npm install socket.io @google/generative-ai langchain @langchain/openai @langchain/core openai dotenv
```

### 2. Configure Environment
The `.env` file has been created with your Hugging Face token:
```env
HF_TOKEN=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
MONGODB_URI=mongodb://localhost:27017/techgium
FASTAPI_URL=http://localhost:8000
PORT=5000
```

### 3. Start MongoDB
```bash
# Make sure MongoDB is running
mongod
```

### 4. Start Backend Server
```bash
cd backend
npm run dev
```

---

## 🧪 Testing the Chatbot

### Test 1: Standalone Mode (General Medical Q&A)

**Client-side code:**
```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

// Initialize session
socket.emit('init_session', {
    mode: 'standalone'
});

socket.on('session_initialized', (data) => {
    console.log(data.message);
    
    // Send a message in Hindi
    socket.emit('send_message', {
        text: 'Mujhe sar dard hai, kya karu?',
        language: 'hi'
    });
});

socket.on('receive_message', (data) => {
    console.log('Response:', data.text);
    console.log('Language:', data.language);
    console.log('Emergency:', data.isEmergency);
});

socket.on('emergency_alert', (data) => {
    alert(data.message);
});
```

### Test 2: Context-Aware Mode (With Patient Report)

```javascript
// Initialize with patient ID
socket.emit('init_session', {
    userId: 'user123',
    patientId: 'patient456',
    mode: 'context-aware'
});

socket.on('session_initialized', (data) => {
    if (data.hasContext) {
        console.log('Medical report loaded!');
        
        // Ask about the report
        socket.emit('send_message', {
            text: 'What does my heart rate indicate?',
            language: 'en'
        });
    }
});
```

### Test 3: Language Switching

```javascript
// Change to Hindi
socket.emit('change_language', { language: 'hi' });

socket.on('language_changed', (data) => {
    console.log(data.message); // "भाषा बदल दी गई है"
});
```

---

## 🔄 Pipeline Flow Example

**User Input (Hindi):** "Mujhe bukhar hai aur sar dard hai"

**Pipeline 1 (Translation):**
- Input: "Mujhe bukhar hai aur sar dard hai"
- Output: "I have fever and headache"

**Pipeline 2 (Medical Reasoning):**
- Input: "I have fever and headache"
- Output: "Fever and headache can be symptoms of various conditions including flu, viral infections, or dehydration. Here are some recommendations: 1. Rest adequately 2. Stay hydrated 3. Take over-the-counter pain relievers if needed. If symptoms persist for more than 3 days or worsen, please consult a doctor."

**Pipeline 3 (Translation):**
- Input: "Fever and headache can be symptoms..."
- Output: "बुखार और सिरदर्द विभिन्न स्थितियों के लक्षण हो सकते हैं..."

---

## 📊 Socket.IO Events Reference

### Client → Server Events

| Event | Data | Description |
|-------|------|-------------|
| `init_session` | `{ userId, patientId, mode }` | Initialize chat session |
| `send_message` | `{ text, language }` | Send a message |
| `change_language` | `{ language }` | Change language preference |
| `switch_mode` | `{ mode, patientId }` | Switch between standalone/context-aware |
| `get_history` | - | Get conversation history |
| `clear_history` | - | Clear conversation |

### Server → Client Events

| Event | Data | Description |
|-------|------|-------------|
| `session_initialized` | `{ success, mode, message, hasContext }` | Session ready |
| `receive_message` | `{ text, sender, language, isEmergency, timestamp }` | AI response |
| `emergency_alert` | `{ message, severity }` | Emergency detected |
| `typing` | `{ isTyping }` | Typing indicator |
| `language_changed` | `{ language, message }` | Language updated |
| `mode_switched` | `{ success, mode, message }` | Mode changed |
| `history_loaded` | `{ messages, metadata }` | Conversation history |
| `error` | `{ message, error }` | Error occurred |

---

## 🎯 Features Implemented

✅ **Dual Mode Support**
- Standalone: General medical Q&A
- Context-aware: Q&A with patient's medical report

✅ **Multilingual Support**
- Hindi ↔ English translation
- Auto language detection
- Hinglish support

✅ **Emergency Detection**
- Keyword-based detection
- Real-time alerts
- Admin notifications

✅ **Conversation Management**
- MongoDB persistence
- Session history
- Context retention

✅ **3-Pipeline Architecture**
- Translation → Medical → Translation
- Seamless language handling
- Medical accuracy

---

## 🔧 Next Steps

### Frontend Integration
1. Create React chat component (see example above)
2. Add Socket.IO client connection
3. Implement UI for:
   - Message display
   - Language selector
   - Mode switcher
   - Emergency alerts

### FastAPI Integration
1. Update `fetchPatientReport()` in MedicalChatService.js
2. Point to actual FastAPI endpoint
3. Test report context fetching

### Testing
1. Test Hindi input/output
2. Test English input/output
3. Test Hinglish input
4. Test emergency detection
5. Test patient report context

---

## 📝 Notes

- **Models Used**: 
  - Medical: `openai/gpt-oss-120b:groq`
  - Translation: `CohereLabs/command-a-translate-08-2025:cohere`
- **API**: Hugging Face Router (https://router.huggingface.co/v1)
- **Token**: Already configured in `.env`
- **Database**: MongoDB (conversation history)

---

## 🐛 Troubleshooting

**Issue**: "Failed to get medical response"
- **Solution**: Check HF_TOKEN in `.env` file
- **Solution**: Verify Hugging Face API is accessible

**Issue**: "Medical report not found"
- **Solution**: Ensure FastAPI is running
- **Solution**: Check FASTAPI_URL in `.env`

**Issue**: Translation not working
- **Solution**: Check if Cohere model is accessible
- **Solution**: Verify input text encoding

---

**Ready to test!** Start the backend server and connect from your React client. 🚀

import mongoose from "mongoose";

const conversationSchema = new mongoose.Schema({
    sessionId: {
        type: String,
        required: true,
        unique: true,
        index: true
    },
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        default: null
    },
    patientId: {
        type: String,
        default: null
    },
    mode: {
        type: String,
        enum: ['standalone', 'context-aware'],
        default: 'standalone'
    },
    messages: [{
        role: {
            type: String,
            enum: ['user', 'assistant', 'system'],
            required: true
        },
        content: {
            type: String,
            required: true
        },
        contentEnglish: {
            type: String,
            default: null // Stores English translation for embedding
        },
        language: {
            type: String,
            enum: ['en', 'hi'],
            default: 'en'
        },
        timestamp: {
            type: Date,
            default: Date.now
        },
        isEmergency: {
            type: Boolean,
            default: false
        },
        // Vector embedding for RAG (384-dimensional)
        embedding: {
            type: [Number],
            default: null
        }
    }],
    medicalContext: {
        type: String,
        default: null
    },
    metadata: {
        startedAt: {
            type: Date,
            default: Date.now
        },
        lastActivity: {
            type: Date,
            default: Date.now
        },
        totalMessages: {
            type: Number,
            default: 0
        },
        emergencyCount: {
            type: Number,
            default: 0
        }
    }
}, {
    timestamps: true
});

// Update lastActivity on save
conversationSchema.pre('save', function (next) {
    this.metadata.lastActivity = new Date();
    this.metadata.totalMessages = this.messages.length;
    this.metadata.emergencyCount = this.messages.filter(m => m.isEmergency).length;
    next();
});

// Index for efficient queries
conversationSchema.index({ 'metadata.lastActivity': -1 });
conversationSchema.index({ userId: 1, 'metadata.lastActivity': -1 });

const Conversation = mongoose.model('Conversation', conversationSchema);

export default Conversation;

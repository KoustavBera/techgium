import mongoose from "mongoose";

/**
 * Medical Knowledge Base Schema
 * Stores medical information as embeddings for RAG retrieval
 */
const medicalKnowledgeSchema = new mongoose.Schema({
    // Unique identifier for the knowledge entry
    knowledgeId: {
        type: String,
        required: true,
        unique: true,
        index: true
    },
    // Category of medical knowledge
    category: {
        type: String,
        enum: ['symptom', 'condition', 'treatment', 'medication', 'general', 'emergency'],
        required: true,
        index: true
    },
    // Original content in English
    content: {
        type: String,
        required: true
    },
    // Content summary for quick retrieval
    summary: {
        type: String,
        required: true
    },
    // Vector embedding (384-dimensional for all-MiniLM-L6-v2)
    embedding: {
        type: [Number],
        required: true,
        validate: {
            validator: function (v) {
                return v.length === 384;
            },
            message: 'Embedding must be 384-dimensional'
        }
    },
    // Keywords for fallback search
    keywords: [{
        type: String,
        index: true
    }],
    // Source of the information
    source: {
        type: String,
        default: 'internal'
    },
    // Metadata
    metadata: {
        createdAt: {
            type: Date,
            default: Date.now
        },
        updatedAt: {
            type: Date,
            default: Date.now
        },
        usageCount: {
            type: Number,
            default: 0
        }
    }
}, {
    timestamps: true
});

// Index for vector search (will be created in MongoDB Atlas)
// Note: Atlas Vector Search index must be created manually in Atlas UI
medicalKnowledgeSchema.index({ category: 1, 'metadata.usageCount': -1 });

const MedicalKnowledge = mongoose.model('MedicalKnowledge', medicalKnowledgeSchema);

export default MedicalKnowledge;

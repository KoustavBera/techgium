import { OpenAI } from "openai";
import dotenv from "dotenv";
import MedicalKnowledge from "../models/medicalKnowledge.model.js";
import Conversation from "../models/conversation.model.js";

dotenv.config();

/**
 * Vector Store Service for RAG Implementation
 * Handles embedding generation and similarity search using MongoDB Atlas Vector Search
 */
class VectorStoreService {
    constructor() {
        // Hugging Face client for embeddings
        this.client = new OpenAI({
            baseURL: "https://router.huggingface.co/v1",
            apiKey: process.env.HF_TOKEN,
        });

        // Embedding model configuration
        this.embeddingModel = "sentence-transformers/all-MiniLM-L6-v2";
        this.embeddingDimension = 384;
    }

    /**
     * Generate embedding for text using Hugging Face API
     * @param {string} text - Text to embed
     * @returns {Promise<number[]>} - 384-dimensional embedding vector
     */
    async generateEmbedding(text) {
        try {
            // Use the Hugging Face Inference API for embeddings
            const response = await fetch(
                `https://api-inference.huggingface.co/pipeline/feature-extraction/${this.embeddingModel}`,
                {
                    method: "POST",
                    headers: {
                        "Authorization": `Bearer ${process.env.HF_TOKEN}`,
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        inputs: text,
                        options: { wait_for_model: true }
                    })
                }
            );

            if (!response.ok) {
                throw new Error(`Embedding API error: ${response.status}`);
            }

            const embedding = await response.json();

            // Handle nested array response
            if (Array.isArray(embedding) && Array.isArray(embedding[0])) {
                return embedding[0];
            }

            return embedding;
        } catch (error) {
            console.error("[VectorStore] Embedding error:", error);
            throw error;
        }
    }

    /**
     * Store conversation message with embedding
     * @param {string} sessionId - Session ID
     * @param {string} content - Message content
     * @param {string} contentEnglish - English translation
     * @param {string} role - 'user' or 'assistant'
     * @param {Object} metadata - Additional metadata
     */
    async storeMessageWithEmbedding(sessionId, content, contentEnglish, role, metadata = {}) {
        try {
            // Generate embedding from English content
            const embedding = await this.generateEmbedding(contentEnglish);

            // Find and update conversation
            const conversation = await Conversation.findOne({ sessionId });
            if (conversation) {
                conversation.messages.push({
                    role,
                    content,
                    contentEnglish,
                    language: metadata.language || 'en',
                    embedding,
                    isEmergency: metadata.isEmergency || false,
                    timestamp: new Date()
                });
                await conversation.save();
            }

            return embedding;
        } catch (error) {
            console.error("[VectorStore] Store message error:", error);
            // Store without embedding if embedding fails
            return null;
        }
    }

    /**
     * Search for similar past conversations using vector similarity
     * @param {string} queryText - Query text in English
     * @param {number} limit - Number of results to return
     * @param {string} userId - Optional user ID to filter results
     * @returns {Promise<Array>} - Similar conversation messages
     */
    async searchSimilarConversations(queryText, limit = 5, userId = null) {
        try {
            // Generate query embedding
            const queryEmbedding = await this.generateEmbedding(queryText);

            // Build aggregation pipeline for vector search
            const pipeline = [
                {
                    $unwind: "$messages"
                },
                {
                    $match: {
                        "messages.embedding": { $exists: true, $ne: null },
                        "messages.role": "assistant" // Search in assistant responses
                    }
                }
            ];

            // Add user filter if provided
            if (userId) {
                pipeline.push({
                    $match: { userId: userId }
                });
            }

            // Add vector similarity calculation
            pipeline.push({
                $addFields: {
                    similarity: {
                        $reduce: {
                            input: { $range: [0, { $size: "$messages.embedding" }] },
                            initialValue: 0,
                            in: {
                                $add: [
                                    "$$value",
                                    {
                                        $multiply: [
                                            { $arrayElemAt: ["$messages.embedding", "$$this"] },
                                            { $arrayElemAt: [queryEmbedding, "$$this"] }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                }
            });

            // Sort by similarity and limit
            pipeline.push(
                { $sort: { similarity: -1 } },
                { $limit: limit },
                {
                    $project: {
                        content: "$messages.content",
                        contentEnglish: "$messages.contentEnglish",
                        role: "$messages.role",
                        similarity: 1,
                        timestamp: "$messages.timestamp"
                    }
                }
            );

            const results = await Conversation.aggregate(pipeline);
            return results;
        } catch (error) {
            console.error("[VectorStore] Search error:", error);
            return [];
        }
    }

    /**
     * Search medical knowledge base for relevant information
     * @param {string} queryText - Query text in English
     * @param {number} limit - Number of results
     * @returns {Promise<Array>} - Relevant medical knowledge entries
     */
    async searchMedicalKnowledge(queryText, limit = 3) {
        try {
            // Generate query embedding
            const queryEmbedding = await this.generateEmbedding(queryText);

            // Aggregation pipeline for vector search in knowledge base
            const pipeline = [
                {
                    $addFields: {
                        similarity: {
                            $reduce: {
                                input: { $range: [0, { $size: "$embedding" }] },
                                initialValue: 0,
                                in: {
                                    $add: [
                                        "$$value",
                                        {
                                            $multiply: [
                                                { $arrayElemAt: ["$embedding", "$$this"] },
                                                { $arrayElemAt: [queryEmbedding, "$$this"] }
                                            ]
                                        }
                                    ]
                                }
                            }
                        }
                    }
                },
                { $sort: { similarity: -1 } },
                { $limit: limit },
                {
                    $project: {
                        category: 1,
                        content: 1,
                        summary: 1,
                        similarity: 1,
                        keywords: 1
                    }
                }
            ];

            const results = await MedicalKnowledge.aggregate(pipeline);

            // Update usage count for retrieved entries
            const ids = results.map(r => r._id);
            await MedicalKnowledge.updateMany(
                { _id: { $in: ids } },
                { $inc: { "metadata.usageCount": 1 } }
            );

            return results;
        } catch (error) {
            console.error("[VectorStore] Medical knowledge search error:", error);
            return [];
        }
    }

    /**
     * Add medical knowledge to the knowledge base
     * @param {Object} knowledge - Knowledge entry
     * @returns {Promise<Object>} - Created entry
     */
    async addMedicalKnowledge(knowledge) {
        try {
            const { category, content, summary, keywords = [], source = 'internal' } = knowledge;

            // Generate embedding
            const embedding = await this.generateEmbedding(content);

            // Create unique ID
            const knowledgeId = `MK_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

            const entry = new MedicalKnowledge({
                knowledgeId,
                category,
                content,
                summary,
                embedding,
                keywords,
                source
            });

            await entry.save();
            console.log(`[VectorStore] Added knowledge: ${knowledgeId}`);
            return entry;
        } catch (error) {
            console.error("[VectorStore] Add knowledge error:", error);
            throw error;
        }
    }

    /**
     * Build RAG context from retrieved documents
     * @param {string} queryText - English query
     * @param {string} userId - Optional user ID
     * @returns {Promise<string>} - Combined context for LLM
     */
    async buildRAGContext(queryText, userId = null) {
        try {
            // Get similar past conversations
            const pastConversations = await this.searchSimilarConversations(queryText, 3, userId);

            // Get relevant medical knowledge
            const medicalKnowledge = await this.searchMedicalKnowledge(queryText, 2);

            let context = "";

            // Add relevant medical knowledge
            if (medicalKnowledge.length > 0) {
                context += "RELEVANT MEDICAL INFORMATION:\n";
                medicalKnowledge.forEach((k, i) => {
                    context += `${i + 1}. [${k.category.toUpperCase()}] ${k.summary}\n`;
                });
                context += "\n";
            }

            // Add similar past conversations
            if (pastConversations.length > 0) {
                context += "RELEVANT PAST CONVERSATIONS:\n";
                pastConversations.forEach((c, i) => {
                    const contentPreview = c.contentEnglish || c.content;
                    context += `${i + 1}. ${contentPreview.substring(0, 200)}...\n`;
                });
                context += "\n";
            }

            return context;
        } catch (error) {
            console.error("[VectorStore] Build RAG context error:", error);
            return "";
        }
    }
}

// Export singleton
export default new VectorStoreService();

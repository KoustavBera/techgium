import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

export const handleChat = (socket, io) => {
    socket.on("send_message", async (data) => {
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const prompt = `You are a medical assistant. Answer the user's question: ${data.text}. Respond in the user's language.`;
        
        const result = await model.generateContent(prompt);
        const response = await result.response;
        
        socket.emit("receive_message", {
            text: response.text(),
            sender: "assistant"
        });
    });
};
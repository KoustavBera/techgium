import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import dotenv from "dotenv";
import cors from "cors";
import errorHandler from "./utils/errorHandler.js";

import { handleChat } from "./chatbot/chatHandler.js";

dotenv.config();

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
	cors: {
		origin: "*",
		methods: ["GET", "POST"],
	},
});

const PORT = process.env.PORT || 5000;

// --- Middlewares ---
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// --- Routes ---
app.get("/", (req, res) => {
	res.status(200).json({ message: "API is running..." });
});

// --- Socket.io ---
io.on("connection", (socket) => {
	console.log(`User connected: ${socket.id}`);
	handleChat(socket, io);
});

// Error handling
app.use(errorHandler);

// --- Start Server ---
httpServer.listen(PORT, () => {
	console.log(
		`Server running in ${process.env.NODE_ENV || "development"
		} mode on port ${PORT}`
	);
});


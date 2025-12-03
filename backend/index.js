import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import mongoose from "mongoose";
import errorHandler from "./utils/errorHandler.js";
import connectDB from "./config/connectDB.js";

// Load env variables
dotenv.config();

// Initialize App
const app = express();
const PORT = process.env.PORT || 5000;

// --- Middlewares ---
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(errorHandler);

// --- Routes ---

// 1. Health Check
app.get("/", (req, res) => {
	res.status(200).json({ message: "API is running..." });
});

// --- Start Server ---
app.listen(PORT, () => {
	console.log(
		`Server running in ${
			process.env.NODE_ENV || "development"
		} mode on port ${PORT}`
	);
});
connectDB();

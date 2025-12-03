import mongoose from "mongoose";
import asyncHandler from "../utils/asyncHandler.js";

const connectDB = asyncHandler(async () => {
	const conn = await mongoose.connect(process.env.MONGOURI);
	console.log(`MongoDB connected: ${conn.connection.host}`);
});

export default connectDB;

import mongoose from "mongoose";
import asyncHandler from "../utils/asyncHandler.js";

const userSchema = new mongoose.Schema(
	{
		username: {
			type: String,
			required: true,
		},
		phone_no: {
			type: Number,
			required: true,
		},
		lat: {
			type: Number,
			required: true,
		},
		long: {
			type: Number,
			required: true,
		},
		Address: {
			type: String,
			required: true,
		},
		gender: {
			type: String,
			enum: ["Male", "Female", "Other"],
		},
		medicalHistory: {
			type: [String],
			default: [],
		},
		emergencyContact: {
			name: { type: String },
			phone_no: { type: Number },
		},
		visits: [
			{
				date: { type: Date, default: Date.now },
				chamberId: { type: mongoose.Schema.Types.ObjectId, ref: "Chamber" },
				diagnostics: [
					{ type: mongoose.Schema.Types.ObjectId, ref: "DiagnosticReport" },
				],
			},
		],
	},
	{ timestamps: true }
);

export default mongoose.model("User", userSchema);

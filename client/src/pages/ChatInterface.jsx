import React, { useState, useRef, useEffect } from "react";

const ChatInterface = () => {
	const [messages, setMessages] = useState([
		{
			id: 1,
			role: "assistant",
			content:
				"Hello! I'm your AI health assistant. I'll help you through a comprehensive health assessment. How are you feeling today?",
			timestamp: new Date(),
		},
	]);
	const [input, setInput] = useState("");
	const [isTyping, setIsTyping] = useState(false);
	const messagesEndRef = useRef(null);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	};

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

	const sendMessage = () => {
		if (!input.trim()) return;

		const userMessage = {
			id: Date.now(),
			role: "user",
			content: input,
			timestamp: new Date(),
		};

		setMessages((prev) => [...prev, userMessage]);
		setInput("");
		setIsTyping(true);

		// Simulate AI response
		setTimeout(() => {
			const responses = [
				"Thank you for sharing. Can you tell me more about when these symptoms started?",
				"I understand. Let me ask you a few follow-up questions to better assess your condition.",
				"Based on what you've told me, I'd like to gather some additional information.",
				"That's helpful. Can you describe the intensity of your symptoms on a scale of 1-10?",
			];

			const aiResponse = {
				id: Date.now() + 1,
				role: "assistant",
				content: responses[Math.floor(Math.random() * responses.length)],
				timestamp: new Date(),
			};

			setMessages((prev) => [...prev, aiResponse]);
			setIsTyping(false);
		}, 1500);
	};

	const quickResponses = [
		"Yes",
		"No",
		"Not sure",
		"Tell me more",
		"I need help",
	];

	return (
		<div className="space-y-6">
			<div className="flex justify-between items-center">
				<div>
					<h1 className="text-3xl font-bold text-gray-900">
						AI Health Assistant
					</h1>
					<p className="text-sm text-gray-600 mt-1">
						Smart questionnaire for comprehensive health assessment
					</p>
				</div>
				<button className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition-colors">
					🚨 Emergency Help
				</button>
			</div>

			<div
				className="bg-white rounded-xl shadow-sm border border-gray-100 flex flex-col"
				style={{ height: "calc(100vh - 200px)" }}
			>
				{/* Chat Header */}
				<div className="p-4 border-b border-gray-100 bg-blue-50 rounded-t-xl">
					<div className="flex items-center space-x-3">
						<div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
							<span className="text-white text-lg">🤖</span>
						</div>
						<div>
							<h3 className="font-semibold text-gray-900">
								AI Health Assistant
							</h3>
							<p className="text-sm text-gray-600">
								{isTyping ? "Typing..." : "Online • Ready to help"}
							</p>
						</div>
					</div>
				</div>

				{/* Messages */}
				<div className="flex-1 overflow-y-auto p-4 space-y-4">
					{messages.map((message) => (
						<div
							key={message.id}
							className={`flex ${
								message.role === "user" ? "justify-end" : "justify-start"
							}`}
						>
							<div
								className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
									message.role === "user"
										? "bg-blue-600 text-white rounded-br-sm"
										: "bg-gray-100 text-gray-900 rounded-bl-sm"
								}`}
							>
								<p className="text-sm">{message.content}</p>
								<p className="text-xs mt-1 opacity-75">
									{message.timestamp.toLocaleTimeString([], {
										hour: "2-digit",
										minute: "2-digit",
									})}
								</p>
							</div>
						</div>
					))}

					{/* Typing indicator */}
					{isTyping && (
						<div className="flex justify-start">
							<div className="bg-gray-100 rounded-lg px-4 py-3 rounded-bl-sm">
								<div className="flex space-x-1">
									<div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
									<div
										className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
										style={{ animationDelay: "0.1s" }}
									></div>
									<div
										className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
										style={{ animationDelay: "0.2s" }}
									></div>
								</div>
							</div>
						</div>
					)}

					<div ref={messagesEndRef} />
				</div>

				{/* Quick Responses */}
				<div className="px-4 py-2 border-t border-gray-100">
					<div className="flex flex-wrap gap-2">
						{quickResponses.map((response, index) => (
							<button
								key={index}
								onClick={() => setInput(response)}
								className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-blue-100 hover:text-blue-700 transition-colors"
							>
								{response}
							</button>
						))}
					</div>
				</div>

				{/* Input */}
				<div className="p-4 border-t border-gray-100">
					<div className="flex space-x-3">
						<button className="p-2 text-gray-400 hover:text-gray-600 rounded-md hover:bg-gray-100">
							<svg
								className="w-5 h-5"
								fill="none"
								stroke="currentColor"
								viewBox="0 0 24 24"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
								/>
							</svg>
						</button>
						<input
							type="text"
							value={input}
							onChange={(e) => setInput(e.target.value)}
							onKeyPress={(e) => e.key === "Enter" && sendMessage()}
							placeholder="Type your message here..."
							className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
						/>
						<button
							onClick={sendMessage}
							disabled={!input.trim()}
							className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
						>
							Send
						</button>
					</div>
				</div>
			</div>
		</div>
	);
};

export default ChatInterface;

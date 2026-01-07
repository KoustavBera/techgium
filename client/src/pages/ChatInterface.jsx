import React, { useState, useRef, useEffect } from "react";
import { Send, Bot, User, AlertCircle, Sparkles, Clock, MessageSquare, ChevronRight, Activity } from 'lucide-react';
import io from "socket.io-client";

const socket = io("http://localhost:5000"); // Connect to your Node.js backend

const ChatInterface = () => {
	const [messages, setMessages] = useState([
		{
			id: 1,
			role: "assistant",
			content: "Hello! I'm your AI health assistant. I'll help you through a comprehensive health assessment. How are you feeling today?",
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

    // Real-time listener for AI responses
    useEffect(() => {
        socket.on("receive_message", (data) => {
            const aiMessage = {
                id: Date.now(),
                role: "assistant",
                content: data.text,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, aiMessage]);
            setIsTyping(false);
        });

        return () => socket.off("receive_message");
    }, []);

	const sendMessage = () => {
		if (!input.trim()) return;

		const userMessage = {
			id: Date.now(),
			role: "user",
			content: input,
			timestamp: new Date(),
		};

		setMessages((prev) => [...prev, userMessage]);

    socket.emit("send_message", { text: input }); 
        
		setInput("");
		setIsTyping(true); // Show "Processing..." while waiting for socket
	};

	const quickResponses = ["Yes", "No", "Not sure", "Tell me more", "I need help"];

  return (
    <div className="h-[calc(100vh-2rem)] max-w-6xl mx-auto p-4 sm:p-6 lg:p-8 animate-fadeIn font-sans">
      <div className="h-full bg-white/90 backdrop-blur-2xl rounded-[2rem] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.05)] border border-slate-200 flex flex-col overflow-hidden relative ring-1 ring-slate-900/5">
        
        {/* Header */}
        <div className="px-8 py-6 border-b border-slate-100 bg-white/80 backdrop-blur-md flex justify-between items-center z-10 sticky top-0">
          <div className="flex items-center space-x-6">
            <div className="relative group">
              <div className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-500/20 group-hover:scale-105 transition-transform duration-300">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-[3px] border-white drop-shadow-sm" title="Online"></div>
            </div>
            <div>
              <h1 className="text-2xl font-serif font-bold text-slate-900 tracking-tight flex items-center gap-2">
                AI Health Assistant
                <Sparkles className="w-4 h-4 text-amber-400" />
              </h1>
              <div className="flex items-center space-x-2 mt-1">
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <p className="text-sm text-slate-500 font-medium font-sans">
                  {isTyping ? "Processing..." : "Active • HIPAA Compliant"}
                </p>
              </div>
            </div>
          </div>
          <button className="bg-white text-rose-600 px-6 py-3 rounded-xl hover:bg-rose-50 transition-colors font-semibold flex items-center space-x-2 border border-rose-100 shadow-sm group">
            <AlertCircle className="w-5 h-5 group-hover:animate-pulse" />
            <span className="text-sm tracking-wide uppercase">Emergency</span>
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-8 space-y-8 bg-slate-50/50 scroll-smooth">
          {messages.map((message) => (
            <div key={message.id} className={`flex group ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[80%] lg:max-w-[65%] p-6 rounded-[1.5rem] shadow-sm transition-all duration-300 ${
                  message.role === "user"
                    ? "bg-slate-900 text-white rounded-br-sm shadow-xl shadow-slate-900/10"
                    : "bg-white text-slate-700 rounded-bl-sm border border-slate-100 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.03)]"
                }`}>
                <div className="flex items-center gap-2 mb-2 opacity-50 text-xs font-medium tracking-wider uppercase">
                    {message.role === 'user' ? <User className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
                    <span>{message.role === 'user' ? 'You' : 'Assistant'}</span>
                </div>
                <p className="text-[1.05rem] leading-relaxed font-light tracking-wide">{message.content}</p>
                <div className="flex items-center justify-end gap-1 mt-3 opacity-60 text-slate-400">
                    <Clock className="w-3 h-3" />
                    <p className="text-xs font-medium tracking-wider uppercase">
                    {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </p>
                </div>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start animate-fadeIn">
              <div className="bg-white rounded-[1.5rem] rounded-bl-sm px-6 py-5 border border-slate-100 shadow-sm flex items-center gap-3">
                <Bot className="w-5 h-5 text-indigo-500 animate-pulse" />
                <div className="flex space-x-1.5">
                  {[0, 150, 300].map((delay, i) => (
                    <div key={i} className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: `${delay}ms` }}></div>
                  ))}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-6 bg-white border-t border-slate-100">
          <div className="flex flex-wrap gap-2 mb-4">
            {quickResponses.map((response, index) => (
              <button key={index} onClick={() => setInput(response)} className="px-4 py-2 text-sm font-medium bg-slate-50 text-slate-600 rounded-full hover:bg-white hover:text-indigo-600 hover:shadow-md hover:ring-1 hover:ring-indigo-100 border border-slate-200 transition-all duration-300 transform hover:-translate-y-0.5 flex items-center gap-2">
                <span>{response}</span>
                <ChevronRight className="w-3 h-3 opacity-0 group-hover:opacity-100" />
              </button>
            ))}
          </div>

          <div className="relative group">
            <div className="relative flex items-center bg-slate-50 rounded-2xl shadow-inner border border-slate-200 p-2 focus-within:bg-white focus-within:ring-2 focus-within:ring-indigo-500/20 focus-within:border-indigo-500 transition-all duration-300">
              <button className="p-4 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-xl transition-all duration-200">
                <Activity className="w-6 h-6" />
              </button>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Describe your symptoms..."
                className="flex-1 px-4 py-3 bg-transparent focus:outline-none text-slate-900 font-medium"
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim()}
                className="ml-2 bg-indigo-600 text-white px-8 py-3 rounded-xl hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg font-bold tracking-wide active:scale-95 flex items-center gap-2"
              >
                <span>Send</span>
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
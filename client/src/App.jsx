import React, { useState } from "react";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import ChatInterface from "./pages/ChatInterface";
import History from "./pages/History";
import Reports from "./pages/Reports";
import Profile from "./pages/Profile";
import Settings from "./pages/Settings";

function App() {
	const [activeTab, setActiveTab] = useState("dashboard");
	const [sidebarOpen, setSidebarOpen] = useState(false);

	const renderPage = () => {
		switch (activeTab) {
			case "dashboard":
				return <Dashboard />;
			case "chat":
				return <ChatInterface />;
			case "history":
				return <History />;
			case "reports":
				return <Reports />;
			case "profile":
				return <Profile />;
			case "settings":
				return <Settings />;
			default:
				return <Dashboard />;
		}
	};

	return (
		<Layout
			activeTab={activeTab}
			setActiveTab={setActiveTab}
			sidebarOpen={sidebarOpen}
			setSidebarOpen={setSidebarOpen}
		>
			{renderPage()}
		</Layout>
	);
}

export default App;

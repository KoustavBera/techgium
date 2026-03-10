import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import HomePage from './pages/HomePage'
import ChatPage from './pages/ChatPage'
import ScreeningPage from './pages/ScreeningPage'

// Phase 4 + 5 pages — placeholder until those phases are built
const Placeholder = ({ name }) => (
  <div
    style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      gap: '12px',
      textAlign: 'center',
    }}
  >
    <h2
      style={{
        fontFamily: "'Google Sans', sans-serif",
        fontSize: '28px',
        color: 'var(--md-primary)',
      }}
    >
      {name}
    </h2>
    <p style={{ color: 'var(--md-on-surface-variant)', fontSize: '15px' }}>
      Coming in the next phase…
    </p>
  </div>
)

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<HomePage />} />
          <Route path="screening" element={<ScreeningPage />} />
          <Route path="chatbot" element={<ChatPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App

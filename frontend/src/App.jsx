import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/authContext'
import Landing from './components/landing'
import Login from './components/auth/login'
import Register from './components/auth/register'
import Home from './components/home'
import './App.css'

function App() {
  const { userLoggedIn } = useAuth()

  return (
    <div className="App">
      <Routes>
        {/* Landing page - public route */}
        <Route path="/" element={<Landing />} />
        
        {/* Auth routes */}
        <Route path="/login" element={!userLoggedIn ? <Login /> : <Navigate to="/home" replace />} />
        <Route path="/register" element={!userLoggedIn ? <Register /> : <Navigate to="/home" replace />} />
        
        {/* Protected routes */}
        <Route path="/home" element={userLoggedIn ? <Home /> : <Navigate to="/login" replace />} />
        <Route path="/dashboard" element={userLoggedIn ? <Home /> : <Navigate to="/login" replace />} />
      </Routes>
    </div>
  )
}

export default App


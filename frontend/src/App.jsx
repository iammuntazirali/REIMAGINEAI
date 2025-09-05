import { Routes, Route, Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard/Dashboard'
import AuthForm from './components/Auth/AuthForm'
import UserTypeSelector from './components/Auth/UserTypeSelector'
import { getStoredUser, clearAuth } from './services/api'
import './App.css'

function Navbar() {
  return (
    <div className="nav">
      <Link to="/" className="brand">REIMAGINE AI</Link>
      <div className="nav-actions">
        <Link to="/auth" className="btn ghost">Login</Link>
        <Link to="/auth" className="btn solid">Sign up</Link>
      </div>
    </div>
  )
}

function Landing() {
  return (
    <div className="landing">
      <Navbar />
      <div className="bg" aria-hidden="true">
        <div className="bg-orb orb-a" />
        <div className="bg-orb orb-b" />
        <div className="bg-orb orb-c" />
      </div>
      <div className="hero">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="title"
        >
          REIMAGINE AI
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.1 }}
          className="subtitle"
        >
          Imagine. Create. Evolve. A modern canvas turning ideas into vivid realities.
        </motion.p>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.2, delay: 0.2 }}
          className="cta"
        >
          <Link to="/auth" className="btn primary">Get started</Link>
          <a href="#showcase" className="btn ghost">See showcase</a>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.2, delay: 0.25 }}
          className="hero-art"
        >
          <div className="gradient orb orb-1" />
          <div className="gradient orb orb-2" />
          <div className="gradient orb orb-3" />
        </motion.div>
      </div>

      <section id="showcase" className="section">
        <motion.h2 initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
          Crafted with intention
        </motion.h2>
        <motion.div
          className="cards"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-80px' }}
          variants={{
            hidden: { opacity: 0, y: 20 },
            visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.12 } }
          }}
        >
          {["Generative Studio", "Prompt Orchestrator", "Realtime Render"].map((title, i) => (
            <motion.div key={title} className="card feature" variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}>
              <div className="card-inner">
                <div className="badge">0{i+1}</div>
                <h3>{title}</h3>
                <p>Beautifully smooth interactions, purposeful motion, and a focus on craft.</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      <footer className="footer">
        <span>© {new Date().getFullYear()} REIMAGINE AI</span>
        <a href="#" className="link">Brand Manifesto</a>
      </footer>
    </div>
  )
}

function Auth() {
  const [selectedUserType, setSelectedUserType] = useState(null)
  const [user, setUser] = useState(null)
  const navigate = useNavigate()

  const handleUserTypeSelect = (userType) => {
    setSelectedUserType(userType)
  }

  const handleLogin = (userData) => {
    setUser(userData)
    navigate('/dashboard')
  }

  const handleLogout = () => {
    setUser(null)
    setSelectedUserType(null)
    navigate('/')
  }

  if (!selectedUserType) {
    return <UserTypeSelector onSelectUserType={handleUserTypeSelect} />
  }

  return <AuthForm userType={selectedUserType} onLogin={handleLogin} />
}

function DashboardPage() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    // Check for stored user data
    const storedUser = getStoredUser()
    if (storedUser) {
      setUser(storedUser)
    } else {
      // No user data, redirect to auth
      navigate('/auth')
    }
    setLoading(false)
  }, [navigate])

  const handleLogout = () => {
    clearAuth()
    setUser(null)
    navigate('/')
  }

  if (loading) {
    return (
      <div className="dashboard-wrapper">
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          color: '#fff',
          fontSize: '1.2rem'
        }}>
          Loading...
        </div>
      </div>
    )
  }

  if (!user) {
    return null // Will redirect to auth
  }

  return (
    <div className="dashboard-wrapper">
      <Dashboard userType={user.userType} onLogout={handleLogout} />
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/auth" element={<Auth />} />
      <Route path="/dashboard" element={<DashboardPage />} />
    </Routes>
  )
}

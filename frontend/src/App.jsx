import { Routes, Route, Navigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import './App.css'

import Login from "./components/auth/login"
import Register from "./components/auth/register"
import Header from "./components/header"
import Home from "./components/home"
import Dashboard from './components/Dashboard'
import PostCard from './components/Postcard'

import { AuthProvider, useAuth } from "./contexts/authContext"

function Navbar() {
  const { userLoggedIn, currentUser } = useAuth()
  
  return (
    <div className="nav">
      <div className="brand">REIMAGINE AI</div>
      <div className="nav-actions">
        {!userLoggedIn ? (
          <>
            <a href="/login" className="btn ghost">Login</a>
            <a href="/register" className="btn solid">Sign up</a>
          </>
        ) : (
          <span className="text-white">Welcome, {currentUser?.email}</span>
        )}
      </div>
    </div>
  )
}

function Landing() {
  const { userLoggedIn } = useAuth()
  
  if (userLoggedIn) {
    return <Navigate to="/dashboard" replace={true} />
  }
  
  return (
    <div className="landing">
      <Navbar />
      <div className="hero">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="title"
        >
          Reimagine what AI can create
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.1 }}
          className="subtitle"
        >
          A modern AI canvas that turns your ideas into vivid realities.
        </motion.p>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.2, delay: 0.2 }}
          className="cta"
        >
          <a href="/login" className="btn primary">Get started</a>
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
          Intelligent AI Solutions
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
          {[
            {
              title: "Toxicity Detection",
              description: "Advanced AI models detect and blur harmful content, creating safer online communities through real-time content moderation."
            },
            {
              title: "Smart Summarization", 
              description: "Intelligent content summarization that extracts key insights from lengthy posts and discussions for quick comprehension."
            },
            {
              title: "Content Recommendations",
              description: "Personalized AI-powered recommendations that surface relevant content based on user interests and engagement patterns."
            }
          ].map((feature, i) => (
            <motion.div key={feature.title} className="card feature" variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}>
              <div className="card-inner">
                <div className="badge">0{i+1}</div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
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

function AppContent() {
  const { userLoggedIn } = useAuth();
  
  return (
    <div className="w-full h-screen flex flex-col">
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={!userLoggedIn ? <Login /> : <Navigate to="/dashboard" replace />} />
        <Route path="/register" element={!userLoggedIn ? <Register /> : <Navigate to="/dashboard" replace />} />
        <Route path="/home" element={userLoggedIn ? <Home /> : <Navigate to="/login" replace />} />
        <Route path="/dashboard" element={userLoggedIn ? <Dashboard /> : <Navigate to="/login" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

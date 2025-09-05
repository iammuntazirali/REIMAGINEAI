import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../contexts/authContext'
import './landing.css'

const Landing = () => {
  const { userLoggedIn } = useAuth()
  const navigate = useNavigate()
  const [isLoaded, setIsLoaded] = useState(false)
  const [currentText, setCurrentText] = useState(0)

  const animatedTexts = [
    "Transform Ideas into Reality",
    "Reimagine Creative Possibilities", 
    "Unleash AI-Powered Innovation",
    "Create Beyond Imagination"
  ]

  useEffect(() => {
    setIsLoaded(true)
    const interval = setInterval(() => {
      setCurrentText((prev) => (prev + 1) % animatedTexts.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const handleGetStarted = () => {
    if (userLoggedIn) {
      navigate('/home')
    } else {
      navigate('/login')
    }
  }

  return (
    <div className="landing-container">
      {/* Navigation */}
      <nav className="landing-nav">
        <div className="nav-brand">
          <div className="brand-logo">
            <div className="logo-icon">
              <div className="ai-symbol"></div>
            </div>
            <span className="brand-name">REIMAGINE AI</span>
          </div>
        </div>
        
        <div className="nav-auth">
          {userLoggedIn ? (
            <button onClick={() => navigate('/home')} className="nav-btn dashboard-btn">
              Dashboard
            </button>
          ) : (
            <>
              <Link to="/login" className="nav-btn login-btn">Login</Link>
              <Link to="/register" className="nav-btn signup-btn">Sign Up</Link>
            </>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <main className={`hero-section ${isLoaded ? 'loaded' : ''}`}>
        <div className="hero-background">
          <div className="gradient-orb orb-1"></div>
          <div className="gradient-orb orb-2"></div>
          <div className="gradient-orb orb-3"></div>
          <div className="grid-overlay"></div>
        </div>

        <div className="hero-content">
          <div className="hero-badge">
            <span className="badge-text">✨ Next-Gen AI Technology</span>
          </div>
          
          <h1 className="hero-title">
            <span className="title-main">REIMAGINE</span>
            <span className="title-highlight">AI</span>
          </h1>
          
          <div className="hero-subtitle">
            <p className="animated-text">
              {animatedTexts[currentText]}
            </p>
          </div>
          
          <p className="hero-description">
            Harness the power of artificial intelligence to transform your creative process. 
            From concept to creation, our AI understands your vision and brings it to life 
            with unprecedented precision and innovation.
          </p>

          <div className="hero-actions">
            <button onClick={handleGetStarted} className="cta-primary">
              <span>Get Started</span>
              <div className="btn-glow"></div>
            </button>
            <button className="cta-secondary">
              <span>Explore Features</span>
            </button>
          </div>

          <div className="hero-stats">
            <div className="stat-item">
              <div className="stat-number">10K+</div>
              <div className="stat-label">Creations Generated</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">99.9%</div>
              <div className="stat-label">Accuracy Rate</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">24/7</div>
              <div className="stat-label">AI Availability</div>
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <div className="ai-interface">
            <div className="interface-card card-1">
              <div className="card-header">
                <div className="status-dot active"></div>
                <span>AI Processing</span>
              </div>
              <div className="processing-bars">
                <div className="bar bar-1"></div>
                <div className="bar bar-2"></div>
                <div className="bar bar-3"></div>
              </div>
            </div>
            
            <div className="interface-card card-2">
              <div className="card-content">
                <div className="neural-network">
                  <div className="node node-1"></div>
                  <div className="node node-2"></div>
                  <div className="node node-3"></div>
                  <div className="connection con-1"></div>
                  <div className="connection con-2"></div>
                </div>
              </div>
            </div>
            
            <div className="interface-card card-3">
              <div className="output-preview">
                <div className="preview-line"></div>
                <div className="preview-line short"></div>
                <div className="preview-line medium"></div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Features Preview */}
      <section className="features-preview">
        <div className="container">
          <h2 className="features-title">Powered by Advanced AI</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon icon-1">🧠</div>
              <h3>Neural Processing</h3>
              <p>Advanced neural networks that understand context and creativity</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon icon-2">⚡</div>
              <h3>Lightning Fast</h3>
              <p>Generate results in seconds with optimized AI algorithms</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon icon-3">🎨</div>
              <h3>Creative Intelligence</h3>
              <p>AI that adapts to your style and enhances your vision</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Landing
import React, { useState } from 'react'
import { useAuth } from '../../contexts/authContext'
import { doSignOut } from '../../firebase/auth'
import { useNavigate } from 'react-router-dom'
import './dashboard.css'

const Home = () => {
    const { currentUser } = useAuth()
    const navigate = useNavigate()
    const [activeTab, setActiveTab] = useState('recommendation')
    const [searchQuery, setSearchQuery] = useState('')
    const [showUserMenu, setShowUserMenu] = useState(false)
    const [isAnalyzing, setIsAnalyzing] = useState(false)

    const handleLogout = async () => {
        try {
            await doSignOut()
            navigate('/login')
        } catch (error) {
            console.error('Logout failed:', error)
        }
    }

    const handleAnalysis = (type) => {
        setIsAnalyzing(true)
        // Simulate analysis
        setTimeout(() => {
            setIsAnalyzing(false)
            setActiveTab('recommendation')
        }, 2000)
    }

    return (
        <div className="dashboard">
            {/* Top Navigation Bar */}
            <nav className="dashboard-nav">
                <div className="nav-left">
                    <h1 className="brand">AI Dashboard</h1>
                    <div className="nav-features">
                        <button 
                            className={`nav-btn ${activeTab === 'toxicity' ? 'active' : ''}`}
                            onClick={() => handleAnalysis('toxicity')}
                            disabled={isAnalyzing}
                        >
                            <span className="btn-icon">🛡️</span>
                            Toxicity Detection
                        </button>
                        <button 
                            className={`nav-btn ${activeTab === 'sentiment' ? 'active' : ''}`}
                            onClick={() => handleAnalysis('sentiment')}
                            disabled={isAnalyzing}
                        >
                            <span className="btn-icon">😊</span>
                            Sentiment Analysis
                        </button>
                    </div>
                </div>
                
                <div className="nav-right">
                    <div className="search-container">
                        <input
                            type="text"
                            placeholder="Search content..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="search-input"
                        />
                        <button className="search-btn">
                            <span className="search-icon">🔍</span>
                        </button>
                    </div>
                    
                    <div className="user-menu">
                        <button 
                            className="user-avatar"
                            onClick={() => setShowUserMenu(!showUserMenu)}
                        >
                            <span className="avatar-text">
                                {currentUser?.email?.charAt(0).toUpperCase()}
                            </span>
                        </button>
                        
                        {showUserMenu && (
                            <div className="user-dropdown">
                                <div className="user-info">
                                    <p className="user-name">
                                        {currentUser?.displayName || 'User'}
                                    </p>
                                    <p className="user-email">{currentUser?.email}</p>
                                </div>
                                <button className="logout-btn" onClick={handleLogout}>
                                    <span className="logout-icon">🚪</span>
                                    Logout
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </nav>

            {/* Main Content Area */}
            <div className="dashboard-content">
                {/* Left Side - Recommendation Section */}
                <div className="main-section">
                    <div className="section-header">
                        <h2 className="section-title">
                            <span className="title-icon">📈</span>
                            Recommendations
                        </h2>
                        <div className="section-actions">
                            <button className="refresh-btn">
                                <span className="refresh-icon">🔄</span>
                                Refresh
                            </button>
                        </div>
                    </div>
                    
                    <div className="recommendation-content">
                        {isAnalyzing ? (
                            <div className="loading-state">
                                <div className="loading-spinner"></div>
                                <p>Analyzing content...</p>
                            </div>
                        ) : (
                            <div className="recommendation-cards">
                                <div className="rec-card">
                                    <div className="card-header">
                                        <span className="card-type">Content Analysis</span>
                                        <span className="card-score positive">+85%</span>
                                    </div>
                                    <h3 className="card-title">Optimize Your Content Strategy</h3>
                                    <p className="card-description">
                                        Based on sentiment analysis, your content shows positive engagement. 
                                        Consider expanding on topics that resonate with your audience.
                                    </p>
                                    <div className="card-tags">
                                        <span className="tag">High Engagement</span>
                                        <span className="tag">Positive Sentiment</span>
                                    </div>
                                </div>
                                
                                <div className="rec-card">
                                    <div className="card-header">
                                        <span className="card-type">Safety Check</span>
                                        <span className="card-score safe">Safe</span>
                                    </div>
                                    <h3 className="card-title">Content Safety Verified</h3>
                                    <p className="card-description">
                                        No toxic content detected. Your content maintains a professional 
                                        and respectful tone throughout.
                                    </p>
                                    <div className="card-tags">
                                        <span className="tag">Toxicity Free</span>
                                        <span className="tag">Professional</span>
                                    </div>
                                </div>
                                
                                <div className="rec-card">
                                    <div className="card-header">
                                        <span className="card-type">Improvement</span>
                                        <span className="card-score neutral">+12%</span>
                                    </div>
                                    <h3 className="card-title">Enhance Readability</h3>
                                    <p className="card-description">
                                        Consider using shorter sentences and simpler vocabulary to 
                                        improve accessibility and engagement.
                                    </p>
                                    <div className="card-tags">
                                        <span className="tag">Readability</span>
                                        <span className="tag">Accessibility</span>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Side - Summarize Section */}
                <div className="sidebar">
                    <div className="summarize-section">
                        <div className="section-header">
                            <h3 className="section-title">
                                <span className="title-icon">📝</span>
                                Summarize
                            </h3>
                        </div>
                        
                        <div className="summarize-content">
                            <div className="input-area">
                                <textarea
                                    placeholder="Paste your content here for analysis and summarization..."
                                    className="content-input"
                                    rows="8"
                                ></textarea>
                                <div className="input-actions">
                                    <button className="analyze-btn">
                                        <span className="btn-icon">⚡</span>
                                        Analyze & Summarize
                                    </button>
                                </div>
                            </div>
                            
                            <div className="summary-output">
                                <h4 className="output-title">Summary</h4>
                                <div className="output-content">
                                    <p>Your content analysis and summary will appear here...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Home
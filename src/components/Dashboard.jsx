import React, { useState, useEffect } from 'react';
import PostCard from "./Postcard.jsx";
import { useAuth } from '../contexts/authContext';
import { doSignOut } from '../firebase/auth';
import { mlService } from '../api/mlservice';
import './home/dashboard.css';

const Dashboard = () => {
  const { currentUser, userLoggedIn } = useAuth();
  const [activeFeature, setActiveFeature] = useState('recommendations');
  const [posts, setPosts] = useState([]);
  const [moderationQueue, setModerationQueue] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [semanticResults, setSemanticResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [summaryText, setSummaryText] = useState('');
  const [selectedPostForSummary, setSelectedPostForSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);

  useEffect(() => {
    if (activeFeature === 'recommendations') fetchRecommendations();
    if (activeFeature === 'posts') fetchPosts();
  }, [activeFeature]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (userDropdownOpen && !event.target.closest('.user-menu')) {
        setUserDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [userDropdownOpen]);

  const fetchPosts = async () => {
    setLoading(true);
    try {
      const res = await mlService.getHotPosts('all', 10);
      setPosts(res.data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const interests = ['AI', 'Technology'];
      const rec = await mlService.recommendPosts(interests);
      console.log(rec)
      setRecommendations(rec.data.data.recommended_items || []);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const flagForModeration = (postId) => {
    if (!moderationQueue.includes(postId)) {
      setModerationQueue(prev => [...prev, postId]);
    }
  };

  const performSemanticSearch = async () => {
    if (!searchQuery) return;
    setLoading(true);
    try {
      const res = await mlService.semanticSearch(searchQuery);
      setSemanticResults(res.data.matches || []);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const summarizePost = async () => {
    if (!selectedPostForSummary) return;
    setLoading(true);
    try {
      const content = selectedPostForSummary.selftext || selectedPostForSummary.title;
      const res = await mlService.getSummary(content);
      setSummaryText(res.data.summary);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const handleLogout = async () => {
    try {
      await doSignOut();
      setUserDropdownOpen(false);
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const getInitials = (email) => {
    if (!email) return 'U';
    const name = email.split('@')[0];
    return name.charAt(0).toUpperCase();
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  return (
    <div className="dashboard">
      <nav className="dashboard-nav">
        <div className="nav-left">
          <h1 className="brand">REIMAGINE AI</h1>
          <div className="nav-features">
            {['recommendations', 'posts', 'toxicity', 'sentiment', 'semantic_search'].map(f => (
              <button
                key={f}
                className={`nav-btn${activeFeature === f ? ' active' : ''}`}
                onClick={() => setActiveFeature(f)}
              >
                {f.replace('_', ' ').toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        
        <div className="nav-right">
          <div className="user-menu">
            <div 
              className="user-avatar"
              onClick={() => setUserDropdownOpen(!userDropdownOpen)}
            >
              {getInitials(currentUser?.email)}
            </div>
            
            {userDropdownOpen && (
              <div className="user-dropdown">
                <div className="user-info">
                  <h3 className="user-name">
                    {currentUser?.displayName || currentUser?.email?.split('@')[0] || 'User'}
                  </h3>
                  <p className="user-email">{currentUser?.email}</p>
                  <div className="user-stats">
                    <div className="stat-item">
                      <span className="stat-label">Member since:</span>
                      <span className="stat-value">{formatDate(currentUser?.metadata?.creationTime)}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Last login:</span>
                      <span className="stat-value">{formatDate(currentUser?.metadata?.lastSignInTime)}</span>
                    </div>
                  </div>
                </div>
                
                <div className="dropdown-actions">
                  <button 
                    className="profile-btn"
                    onClick={() => {
                      setProfileModalOpen(true);
                      setUserDropdownOpen(false);
                    }}
                  >
                    <span className="btn-icon">👤</span>
                    View Profile
                  </button>
                  
                  <button className="logout-btn" onClick={handleLogout}>
                    <span className="logout-icon">🚪</span>
                    Logout
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </nav>

      <main className="dashboard-content">
        {activeFeature === 'recommendations' && (
          <>
            {loading ? <p>Loading Recommendations...</p> : (
              <div className="recommendation-cards">
                {recommendations.map(rec => (
                  <div key={rec.post_id} className="rec-card">
                    <h3>{rec.content}</h3>
                    <p>Score: {rec.score.toFixed(2)}</p>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {activeFeature === 'posts' && (
          <>
            {loading ? <p>Loading Posts...</p> : (
              <div className="recommendation-cards">
                {posts.map(post => (
                  <PostCard key={post.id} post={post} onFlagForModeration={flagForModeration} />
                ))}
              </div>
            )}
          </>
        )}

        {activeFeature === 'toxicity' && (
          <p>Use the posts tab to view toxicity blur and flagged queues.</p>
        )}

        {activeFeature === 'sentiment' && (
          <>
            {/* Sentiment could be shown as a color overlay on posts */}
            <p>Sentiment analysis is visualized in posts tab.</p>
          </>
        )}

        {activeFeature === 'semantic_search' && (
          <>
            <div className="search-container">
              <input
                className="search-input"
                placeholder="Semantic Search..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
              <button className="search-btn" onClick={performSemanticSearch} disabled={loading}>
                Search
              </button>
            </div>
            {loading && <p>Searching...</p>}
            {semanticResults.length > 0 && (
              <div className="recommendation-cards">
                {semanticResults.map((match, idx) => (
                  <div key={idx} className="rec-card">
                    <p>{match.text}</p>
                    <small>Score: {match.score.toFixed(2)}</small>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        <aside className="sidebar">
          <h2>Summarization</h2>
          <textarea
            className="content-input"
            rows={8}
            placeholder="Select a post and click summarize"
            value={selectedPostForSummary ? (selectedPostForSummary.selftext || selectedPostForSummary.title) : ""}
            readOnly
          />
          <button className="analyze-btn" onClick={summarizePost} disabled={!selectedPostForSummary || loading}>
            {loading ? 'Summarizing...' : 'Summarize'}
          </button>
          {summaryText && <div className="summary-output">{summaryText}</div>}
        </aside>
      </main>
      
      {/* User Profile Modal */}
      {profileModalOpen && (
        <div className="modal-overlay" onClick={() => setProfileModalOpen(false)}>
          <div className="profile-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">User Profile</h2>
              <button 
                className="modal-close"
                onClick={() => setProfileModalOpen(false)}
              >
                ×
              </button>
            </div>
            
            <div className="modal-content">
              <div className="profile-avatar-section">
                <div className="profile-avatar">
                  {getInitials(currentUser?.email)}
                </div>
                <div className="avatar-info">
                  <h3 className="profile-name">
                    {currentUser?.displayName || currentUser?.email?.split('@')[0] || 'User'}
                  </h3>
                  <p className="profile-email">{currentUser?.email}</p>
                  <span className={`profile-status ${currentUser?.emailVerified ? 'verified' : 'unverified'}`}>
                    {currentUser?.emailVerified ? '✓ Verified' : '⚠ Unverified'}
                  </span>
                </div>
              </div>
              
              <div className="profile-details">
                <div className="detail-section">
                  <h4 className="section-title">Account Information</h4>
                  <div className="detail-grid">
                    <div className="detail-item">
                      <span className="detail-label">User ID:</span>
                      <span className="detail-value">{currentUser?.uid?.substring(0, 8)}...</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Account Type:</span>
                      <span className="detail-value">
                        {currentUser?.providerData?.[0]?.providerId === 'password' ? 'Email' : 'Social'}
                      </span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Member Since:</span>
                      <span className="detail-value">{formatDate(currentUser?.metadata?.creationTime)}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Last Login:</span>
                      <span className="detail-value">{formatDate(currentUser?.metadata?.lastSignInTime)}</span>
                    </div>
                  </div>
                </div>
                
                <div className="detail-section">
                  <h4 className="section-title">Activity Stats</h4>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <div className="stat-number">24</div>
                      <div className="stat-label">Posts Analyzed</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-number">156</div>
                      <div className="stat-label">Recommendations</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-number">8</div>
                      <div className="stat-label">Summaries Generated</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="modal-actions">
                <button className="btn-secondary" onClick={() => setProfileModalOpen(false)}>
                  Close
                </button>
                <button className="btn-primary" onClick={handleLogout}>
                  <span className="logout-icon">🚪</span>
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;

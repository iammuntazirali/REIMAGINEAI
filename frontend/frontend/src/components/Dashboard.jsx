import React, { useState, useEffect } from 'react';
import PostCard from "./Postcard.jsx";

import { mlService } from '../api/mlservice';
import './home/dashboard.css';

const Dashboard = () => {
  const [activeFeature, setActiveFeature] = useState('recommendations');
  const [posts, setPosts] = useState([]);
  const [moderationQueue, setModerationQueue] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [semanticResults, setSemanticResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [summaryText, setSummaryText] = useState('');
  const [selectedPostForSummary, setSelectedPostForSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (activeFeature === 'recommendations') fetchRecommendations();
    if (activeFeature === 'posts') fetchPosts();
  }, [activeFeature]);

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

  return (
    <div className="dashboard">
      <nav className="dashboard-nav">
        <div className="nav-left">
          <h1 className="brand">Community AI Moderator</h1>
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
    </div>
  );
};

export default Dashboard;

import React, { useState, useEffect } from 'react';
import { mlService } from "../api/mlservice.js";


const PostCard = ({ post, onFlagForModeration }) => {
  const [toxicityScore, setToxicityScore] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [blurContent, setBlurContent] = useState(false);

  useEffect(() => {
    const analyzePost = async () => {
      try {
        const toxRes = await mlService.getToxicity(post.selftext || post.title);
        setToxicityScore(toxRes.data.toxicity_score);
        if (toxRes.data.toxicity_score >= 0.7) setBlurContent(true);
        else if (toxRes.data.toxicity_score >= 0.5) onFlagForModeration(post.id);
      } catch (e) { console.error(e); }
    };
    analyzePost();

    const analyzeSentiment = async () => {
      try {
        const senRes = await mlService.getSentiment(post.selftext || post.title);
        setSentiment(senRes.data.sentiment);
      } catch (e) { console.error(e); }
    };
    analyzeSentiment();
  }, [post, onFlagForModeration]);

  const cardStyle = {
    backgroundColor:
      sentiment === 'positive' ? 'rgba(0,255,0,0.1)' :
      sentiment === 'negative' ? 'rgba(255,0,0,0.1)' : 'transparent',
    filter: blurContent ? 'blur(6px)' : 'none',
    transition: 'all 0.3s ease'
  };

  return (
    <div className="rec-card" style={cardStyle}>
      <h3>{post.title}</h3>
      <p>{post.selftext || '(No content)'}</p>
      <div>
        <small>Toxicity Score: {toxicityScore !== null ? toxicityScore.toFixed(2) : 'Loading...'}</small><br/>
        <small>Sentiment: {sentiment || 'Loading...'}</small>
      </div>
    </div>
  );
};

export default PostCard;

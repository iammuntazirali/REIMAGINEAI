// import apiClient from './config';

// export const mlService = {
//   analyzeContent: (text, postId) =>
//     apiClient.post('/analyze-content', { text, post_id: postId }),
//   getHotPosts: (sub, limit) => apiClient.get(`/reddit/hot/${sub}?limit=${limit}`),
//   searchPosts: (q, sub, limit) =>
//     apiClient.get('/reddit/search', { params:{ query:q, subreddit:sub, limit } }),
// };
import apiClient from "./config.js";

export const mlService = {
  analyzeContent: (text, postId) =>
    apiClient.post('/analyze-content', { text, post_id: postId }),

  getHotPosts: (subreddit, limit = 10) =>
    apiClient.get(`/reddit/hot/${subreddit}`, { params: { limit } }),

  searchPosts: (query, subreddit = 'all', limit = 25) =>
    apiClient.get('/reddit/search', { params: { query, subreddit, limit } }),

  getSentiment: (text) =>
    apiClient.post('/sentiment', { text }),

  getToxicity: (text) =>
    apiClient.post('/toxicity', { text }),

  getSummary: (text) =>
    apiClient.post('/summarize', { text }),

  semanticSearch: (text) =>
    apiClient.post('/semantic-search', { text }),

  recommendPosts: (interests) =>
    apiClient.post('/recommend', { inputs: interests, top_k: 5 }),
};

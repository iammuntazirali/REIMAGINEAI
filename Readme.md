# ReimagineAI: AI-Enhanced Reddit Dashboard

## Overview

ReimagineAI is an innovative dashboard platform that integrates live Reddit community content with advanced AI-powered text analysis and recommendation features. It enables users to explore real-time Reddit posts and comments enriched with intelligent insights such as summarization, sentiment evaluation, toxicity detection, semantic search, and personalized content recommendations — all in one seamless, user-centric interface.

## Unique Selling Points (USP) & Innovations

- **Real-Time Reddit Content:** Fetches live posts and nested comments from Reddit with full metadata.
- **AI-Powered Text Analysis:** Applies state-of-the-art Transformer-based summarization, plus custom models for sentiment and toxicity assessment.
- **Semantic Search & Recommendations:** Delivers personalized and context-aware content discovery informed by semantic embeddings.
- **Unified Dashboard:** Combines Reddit browsing and ML intelligence for a comprehensive, safe, and insightful user experience.
- **Real-Time Updates:** Optional WebSocket integration for live interaction and background processing results.

## Technology Stack

- **Backend:**
  - [FastAPI](https://fastapi.tiangolo.com/) for building high-performance REST and WebSocket APIs.
  - [PRAW](https://praw.readthedocs.io/) for Reddit API integration.
  - [Transformers by Hugging Face](https://huggingface.co/transformers/) for robust text summarization.
  - Custom ML models for sentiment analysis, toxicity scoring, semantic search, and recommendations.
  - Python environment with `dotenv` for environment variable management.

- **Frontend:**
  - [React](https://reactjs.org/) with React Router for SPA routing and secure authenticated routes.
  - React Context API for global authentication state management.
  - Axios for HTTP API communication.
  - WebSocket for real-time ML updates (optional).

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+ and npm
- Reddit developer account credentials (client ID, secret, username, password, user agent)

### Installation

1. Clone the repository:
git clone https://github.com/vermaapurva33/REIMAGINE.git


2. Backend setup:

- Create and activate a Python virtual environment:
  ```
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate     # Windows
  ```

- Install Python dependencies:
  ```
  pip install -r requirements.txt
  ```

- Create a `.env` file in root with your Reddit credentials:
  ```
  REDDIT_CLIENT_ID=your_client_id
  REDDIT_CLIENT_SECRET=your_client_secret
  REDDIT_USERNAME=your_reddit_username
  REDDIT_PASSWORD=your_reddit_password
  REDDIT_USER_AGENT=your_user_agent_string
  FRONTEND_ORIGINS=http://localhost:3000
  ```

3. Frontend setup:

- Navigate to frontend directory:
  ```
  cd frontend
  ```

- Install dependencies:
  ```
  npm install
  ```

- Start frontend development server:
  ```
  npm start
  ```

4. Run backend API server (from project root):

5. Open your browser at [http://localhost:5173](http://localhost:5173) to access the dashboard.

## Project Structure

/
├── backend/ # FastAPI backend
│ ├── main.py # API endpoints and app startup
│ ├── reddit_service.py # Reddit API wrapper using PRAW
│ ├── ml_models/ # ML model code: sentiment, toxicity, summarization, search
│ └── .env # Environment variables (local only)
│
├── frontend/ # React frontend application (Vite-based)
│ ├── src/
│ │ ├── components/ # React components (Dashboard, PostCard, Comments, Auth)
│ │ ├── contexts/ # Auth context/provider
│ │ ├── mlservice.js # Frontend API client wrappers
│ │ └── main.jsx # Vite React app entry point
│ ├── index.html # Vite HTML template
│ └── package.json
│
├── requirements.txt # Backend Python packages
└── README.md # This file



## Features

- Authenticated user login and registration.
- Display of live Reddit posts and nested comments.
- On-demand text summarization using a Transformer-based pipeline.
- Sentiment and toxicity inference to flag harmful content.
- Semantic search for high-precision content retrieval.
- Personalized content recommendations based on user interests.
- Real-time notifications and updates via WebSocket communication.

## Contribution

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request. Ensure code quality and proper testing.

## License

MIT License

---

Built with ❤️ by the ReimagineAI Team  
For inquiries, please contact: vermaapurva33@gmail.com


# REIMAGINE AI - Modern AI Dashboard

A sleek, modern AI dashboard with separate authentication flows for users and moderators, built with React, Node.js, and MongoDB.

## Features

### 🎨 Modern UI/UX Design
- Sleek, aesthetic interface with controlled animations
- GSAP animations for smooth interactions
- Responsive design optimized for mobile and desktop
- Dark theme with gradient accents

### 🔐 Authentication System
- Separate login/signup flows for Users and Moderators
- JWT-based authentication
- Secure password hashing with bcrypt
- Role-based access control

### 📊 Dashboard Features
- Real-time statistics and analytics
- AI model management
- Activity feed
- User management (Moderator only)
- Quick action buttons

### 📱 Mobile Optimized
- Fully responsive design
- Touch-friendly interface
- iOS Safari optimizations
- Mobile-first approach

## Tech Stack

### Frontend
- **React 19** - Modern React with hooks
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first CSS
- **Bootstrap** - Additional UI components
- **Framer Motion** - Smooth animations
- **GSAP** - Advanced animations
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client

### Backend
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **MongoDB** - Database
- **Mongoose** - ODM
- **JWT** - Authentication
- **bcryptjs** - Password hashing
- **CORS** - Cross-origin requests
- **Helmet** - Security headers

## Quick Start

### Prerequisites
- Node.js (v16 or higher)
- MongoDB (local or cloud)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd REIMAGINEAI
   ```

2. **Install dependencies**
   ```bash
   # Install backend dependencies
   cd backend
   npm install
   
   # Install frontend dependencies
   cd ../frontend
   npm install
   ```

3. **Environment Setup**
   - Copy `backend/config.env` and update with your settings
   - Ensure MongoDB is running

4. **Start the application**
   
   **Option 1: Use the startup script (Windows)**
   ```bash
   # From the root directory
   start.bat
   ```
   
   **Option 2: Manual startup**
   ```bash
   # Terminal 1 - Backend
   cd backend
   npm run dev
   
   # Terminal 2 - Frontend
   cd frontend
   npm run dev
   ```

5. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:5000

## Usage

### Authentication Flow

1. **Landing Page**: Visit the homepage to see the modern landing page
2. **User Type Selection**: Choose between User or Moderator access
3. **Registration/Login**: Create an account or sign in
4. **Dashboard**: Access your personalized dashboard

### User Types

#### Regular User
- View dashboard statistics
- Access AI models
- Create and manage projects
- View activity feed

#### Moderator
- All user features
- User management
- System monitoring
- Advanced analytics

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Logout user

### Dashboard
- `GET /api/dashboard/stats` - Get dashboard statistics
- `GET /api/dashboard/activity` - Get recent activity
- `GET /api/dashboard/models` - Get AI models
- `GET /api/dashboard/users` - Get users (moderator only)

### Users
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update user profile
- `GET /api/users` - Get all users (moderator only)

## Project Structure

```
REIMAGINEAI/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Auth/
│   │   │   └── Dashboard/
│   │   ├── services/
│   │   └── App.jsx
│   └── package.json
├── backend/
│   ├── models/
│   ├── routes/
│   ├── middleware/
│   ├── server.js
│   └── package.json
└── start.bat
```

## Development

### Frontend Development
```bash
cd frontend
npm run dev
```

### Backend Development
```bash
cd backend
npm run dev
```

### Building for Production
```bash
# Frontend
cd frontend
npm run build

# Backend
cd backend
npm start
```

## Environment Variables

### Backend (.env)
```
PORT=5000
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
MONGODB_URI=mongodb://localhost:27017/reimagine-ai
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRE=7d
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support, email support@reimagine-ai.com or create an issue in the repository.

---

**REIMAGINE AI** - Imagine. Create. Evolve.

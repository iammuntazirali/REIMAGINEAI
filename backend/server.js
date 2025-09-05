const express = require('express')
const cors = require('cors')
const helmet = require('helmet')
const morgan = require('morgan')
const dotenv = require('dotenv')
const mongoose = require('mongoose')

// Import routes
const authRoutes = require('./routes/auth')
const userRoutes = require('./routes/users')
const dashboardRoutes = require('./routes/dashboard')

// Load environment variables
dotenv.config({ path: './config.env' })

const app = express()
const PORT = process.env.PORT || 5000

// Middleware
app.use(helmet())
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}))
app.use(morgan('combined'))
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true }))

// Database connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/reimagine-ai')
.then(() => console.log('✅ Connected to MongoDB'))
.catch(err => console.error('❌ MongoDB connection error:', err))

// Routes
app.use('/api/auth', authRoutes)
app.use('/api/users', userRoutes)
app.use('/api/dashboard', dashboardRoutes)

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'REIMAGINE AI Backend is running',
    timestamp: new Date().toISOString()
  })
})

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack)
  res.status(500).json({ 
    message: 'Something went wrong!',
    error: process.env.NODE_ENV === 'development' ? err.message : 'Internal server error'
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({ message: 'Route not found' })
})

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`)
  console.log(`📱 Frontend URL: ${process.env.FRONTEND_URL || 'http://localhost:5173'}`)
  console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`)
})

module.exports = app

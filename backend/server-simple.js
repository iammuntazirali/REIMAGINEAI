const express = require('express')
const cors = require('cors')
const helmet = require('helmet')
const morgan = require('morgan')

const app = express()
const PORT = process.env.PORT || 5000

// Middleware
app.use(helmet())
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true
}))
app.use(morgan('combined'))
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true }))

// Mock data
const users = [
  {
    id: 1,
    name: 'John Doe',
    email: 'john@example.com',
    userType: 'user',
    isActive: true,
    lastLogin: new Date().toISOString()
  },
  {
    id: 2,
    name: 'Jane Smith',
    email: 'jane@example.com',
    userType: 'moderator',
    isActive: true,
    lastLogin: new Date().toISOString()
  }
]

// Auth routes
app.post('/api/auth/register', (req, res) => {
  const { name, email, password, userType } = req.body
  
  // Simple validation
  if (!name || !email || !password) {
    return res.status(400).json({
      success: false,
      message: 'Please provide all required fields'
    })
  }

  // Check if user exists
  const existingUser = users.find(u => u.email === email)
  if (existingUser) {
    return res.status(400).json({
      success: false,
      message: 'User with this email already exists'
    })
  }

  // Create new user
  const newUser = {
    id: users.length + 1,
    name,
    email,
    userType: userType || 'user',
    isActive: true,
    lastLogin: new Date().toISOString()
  }
  
  users.push(newUser)

  // Mock token
  const token = 'mock-jwt-token-' + Date.now()

  res.status(201).json({
    success: true,
    message: 'User registered successfully',
    data: {
      user: newUser,
      token
    }
  })
})

app.post('/api/auth/login', (req, res) => {
  const { email, password, userType } = req.body
  
  // Simple validation
  if (!email || !password) {
    return res.status(400).json({
      success: false,
      message: 'Please provide email and password'
    })
  }

  // Find user
  const user = users.find(u => u.email === email)
  if (!user) {
    return res.status(401).json({
      success: false,
      message: 'Invalid credentials'
    })
  }

  // Check user type
  if (userType && user.userType !== userType) {
    return res.status(401).json({
      success: false,
      message: `Invalid credentials for ${userType} account`
    })
  }

  // Mock token
  const token = 'mock-jwt-token-' + Date.now()

  res.json({
    success: true,
    message: 'Login successful',
    data: {
      user,
      token
    }
  })
})

app.get('/api/auth/me', (req, res) => {
  // Mock current user
  const user = users[0]
  res.json({
    success: true,
    data: {
      user
    }
  })
})

// Dashboard routes
app.get('/api/dashboard/stats', (req, res) => {
  const stats = {
    totalProjects: 24,
    activeUsers: 1234,
    aiModels: 8,
    securityScore: 98
  }

  res.json({
    success: true,
    data: {
      stats,
      userType: 'user',
      lastUpdated: new Date().toISOString()
    }
  })
})

app.get('/api/dashboard/activity', (req, res) => {
  const activities = [
    { id: 1, action: 'New AI model deployed', time: '2 minutes ago', type: 'success' },
    { id: 2, action: 'User registration completed', time: '5 minutes ago', type: 'info' },
    { id: 3, action: 'System backup completed', time: '1 hour ago', type: 'success' },
    { id: 4, action: 'Security scan completed', time: '2 hours ago', type: 'warning' }
  ]

  res.json({
    success: true,
    data: {
      activities,
      total: activities.length
    }
  })
})

app.get('/api/dashboard/models', (req, res) => {
  const models = [
    { id: 1, name: 'GPT-4 Vision', status: 'Active', type: 'Text Generation', lastUsed: '2 hours ago' },
    { id: 2, name: 'Claude 3.5', status: 'Active', type: 'Text Generation', lastUsed: '1 hour ago' },
    { id: 3, name: 'DALL-E 3', status: 'Active', type: 'Image Generation', lastUsed: '30 minutes ago' },
    { id: 4, name: 'Midjourney', status: 'Active', type: 'Image Generation', lastUsed: '1 day ago' }
  ]

  res.json({
    success: true,
    data: {
      models,
      total: models.length
    }
  })
})

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'REIMAGINE AI Backend is running',
    timestamp: new Date().toISOString()
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({ message: 'Route not found' })
})

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`)
  console.log(`📱 Frontend URL: http://localhost:5173`)
  console.log(`🌍 Environment: development`)
  console.log(`💡 Using mock data - no database required`)
})

module.exports = app

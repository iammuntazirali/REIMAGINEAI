const express = require('express')
const { authenticate, authorize } = require('../middleware/auth')
const router = express.Router()

// @route   GET /api/dashboard/stats
// @desc    Get dashboard statistics
// @access  Private
router.get('/stats', authenticate, async (req, res) => {
  try {
    const userType = req.user.userType
    
    // Mock data - in a real app, you'd fetch from database
    const stats = {
      user: {
        totalProjects: 24,
        activeUsers: 1234,
        aiModels: 8,
        securityScore: 98
      },
      moderator: {
        totalProjects: 156,
        activeUsers: 1234,
        aiModels: 12,
        securityScore: 99
      }
    }

    const userStats = stats[userType] || stats.user

    res.json({
      success: true,
      data: {
        stats: userStats,
        userType,
        lastUpdated: new Date().toISOString()
      }
    })
  } catch (error) {
    console.error('Dashboard stats error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to fetch dashboard stats',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

// @route   GET /api/dashboard/activity
// @desc    Get recent activity
// @access  Private
router.get('/activity', authenticate, async (req, res) => {
  try {
    // Mock activity data - in a real app, you'd fetch from database
    const activities = [
      {
        id: 1,
        action: 'New AI model deployed',
        time: '2 minutes ago',
        type: 'success',
        user: req.user.name
      },
      {
        id: 2,
        action: 'User registration completed',
        time: '5 minutes ago',
        type: 'info',
        user: req.user.name
      },
      {
        id: 3,
        action: 'System backup completed',
        time: '1 hour ago',
        type: 'success',
        user: req.user.name
      },
      {
        id: 4,
        action: 'Security scan completed',
        time: '2 hours ago',
        type: 'warning',
        user: req.user.name
      }
    ]

    res.json({
      success: true,
      data: {
        activities,
        total: activities.length
      }
    })
  } catch (error) {
    console.error('Dashboard activity error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to fetch activity data',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

// @route   GET /api/dashboard/models
// @desc    Get AI models list
// @access  Private
router.get('/models', authenticate, async (req, res) => {
  try {
    // Mock models data - in a real app, you'd fetch from database
    const models = [
      {
        id: 1,
        name: 'GPT-4 Vision',
        status: 'Active',
        type: 'Text Generation',
        lastUsed: '2 hours ago'
      },
      {
        id: 2,
        name: 'Claude 3.5',
        status: 'Active',
        type: 'Text Generation',
        lastUsed: '1 hour ago'
      },
      {
        id: 3,
        name: 'DALL-E 3',
        status: 'Active',
        type: 'Image Generation',
        lastUsed: '30 minutes ago'
      },
      {
        id: 4,
        name: 'Midjourney',
        status: 'Active',
        type: 'Image Generation',
        lastUsed: '1 day ago'
      }
    ]

    res.json({
      success: true,
      data: {
        models,
        total: models.length
      }
    })
  } catch (error) {
    console.error('Dashboard models error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to fetch models data',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

// @route   GET /api/dashboard/users
// @desc    Get users list (moderator only)
// @access  Private (Moderator)
router.get('/users', authenticate, authorize('moderator', 'admin'), async (req, res) => {
  try {
    // Mock users data - in a real app, you'd fetch from database
    const users = [
      {
        id: 1,
        name: 'John Doe',
        email: 'john@example.com',
        userType: 'user',
        isActive: true,
        lastLogin: '2 hours ago',
        totalProjects: 5
      },
      {
        id: 2,
        name: 'Jane Smith',
        email: 'jane@example.com',
        userType: 'user',
        isActive: true,
        lastLogin: '1 day ago',
        totalProjects: 12
      },
      {
        id: 3,
        name: 'Mike Johnson',
        email: 'mike@example.com',
        userType: 'moderator',
        isActive: true,
        lastLogin: '30 minutes ago',
        totalProjects: 8
      }
    ]

    res.json({
      success: true,
      data: {
        users,
        total: users.length
      }
    })
  } catch (error) {
    console.error('Dashboard users error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to fetch users data',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

module.exports = router

const express = require('express')
const User = require('../models/User')
const { authenticate, authorize } = require('../middleware/auth')
const router = express.Router()

// @route   GET /api/users/profile
// @desc    Get user profile
// @access  Private
router.get('/profile', authenticate, async (req, res) => {
  try {
    res.json({
      success: true,
      data: {
        user: req.user.getPublicProfile()
      }
    })
  } catch (error) {
    console.error('Get profile error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to get user profile',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

// @route   PUT /api/users/profile
// @desc    Update user profile
// @access  Private
router.put('/profile', authenticate, async (req, res) => {
  try {
    const { name, preferences } = req.body
    const updates = {}

    if (name) updates.name = name
    if (preferences) updates.preferences = { ...req.user.preferences, ...preferences }

    const user = await User.findByIdAndUpdate(
      req.user._id,
      updates,
      { new: true, runValidators: true }
    )

    res.json({
      success: true,
      message: 'Profile updated successfully',
      data: {
        user: user.getPublicProfile()
      }
    })
  } catch (error) {
    console.error('Update profile error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to update profile',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

// @route   GET /api/users
// @desc    Get all users (moderator only)
// @access  Private (Moderator)
router.get('/', authenticate, authorize('moderator', 'admin'), async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 10
    const skip = (page - 1) * limit

    const users = await User.find()
      .select('-password')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)

    const total = await User.countDocuments()

    res.json({
      success: true,
      data: {
        users,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit)
        }
      }
    })
  } catch (error) {
    console.error('Get users error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to get users',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

// @route   PUT /api/users/:id/status
// @desc    Update user status (moderator only)
// @access  Private (Moderator)
router.put('/:id/status', authenticate, authorize('moderator', 'admin'), async (req, res) => {
  try {
    const { isActive } = req.body
    const userId = req.params.id

    if (userId === req.user._id.toString()) {
      return res.status(400).json({
        success: false,
        message: 'Cannot modify your own account status'
      })
    }

    const user = await User.findByIdAndUpdate(
      userId,
      { isActive },
      { new: true, runValidators: true }
    ).select('-password')

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      })
    }

    res.json({
      success: true,
      message: `User ${isActive ? 'activated' : 'deactivated'} successfully`,
      data: {
        user: user.getPublicProfile()
      }
    })
  } catch (error) {
    console.error('Update user status error:', error)
    res.status(500).json({
      success: false,
      message: 'Failed to update user status',
      error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    })
  }
})

module.exports = router

import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { gsap } from 'gsap'
import { Eye, EyeOff, User, Lock, Mail, Shield } from 'lucide-react'
import { authAPI, setAuthToken, setStoredUser } from '../../services/api'
import './AuthForm.css'

const AuthForm = ({ userType, onLogin }) => {
  const [isLogin, setIsLogin] = useState(true)
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: ''
  })
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const formRef = useRef(null)

  useEffect(() => {
    // GSAP animations for form entrance
    const tl = gsap.timeline()
    
    tl.fromTo(formRef.current,
      { opacity: 0, scale: 0.9, y: 30 },
      { opacity: 1, scale: 1, y: 0, duration: 0.8, ease: "back.out(1.7)" }
    )
  }, [])

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }))
    }
  }

  const validateForm = () => {
    const newErrors = {}
    
    if (!formData.email) {
      newErrors.email = 'Email is required'
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid'
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required'
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters'
    }
    
    if (!isLogin) {
      if (!formData.name) {
        newErrors.name = 'Name is required'
      }
      
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Please confirm your password'
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match'
      }
    }
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateForm()) return
    
    setLoading(true)
    setErrors({})
    
    try {
      let response
      
      if (isLogin) {
        // Login
        response = await authAPI.login({
          email: formData.email,
          password: formData.password,
          userType
        })
      } else {
        // Register
        response = await authAPI.register({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          userType
        })
      }
      
      if (response.success) {
        // Store token and user data
        setAuthToken(response.data.token)
        setStoredUser(response.data.user)
        
        // Call parent component's login handler
        onLogin({
          userType,
          email: response.data.user.email,
          name: response.data.user.name,
          isLogin,
          user: response.data.user
        })
      }
    } catch (error) {
      console.error('Authentication error:', error)
      
      if (error.response?.data?.message) {
        setErrors({ general: error.response.data.message })
      } else {
        setErrors({ general: 'Authentication failed. Please try again.' })
      }
    } finally {
      setLoading(false)
    }
  }

  const toggleMode = () => {
    setIsLogin(!isLogin)
    setFormData({
      email: '',
      password: '',
      confirmPassword: '',
      name: ''
    })
    setErrors({})
  }

  return (
    <motion.div 
      ref={formRef}
      className="auth-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-icon">
            {userType === 'moderator' ? <Shield className="icon" /> : <User className="icon" />}
          </div>
          <h2 className="auth-title">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h2>
          <p className="auth-subtitle">
            {isLogin 
              ? `Sign in to your ${userType} account` 
              : `Join as a ${userType}`
            }
          </p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {!isLogin && (
            <motion.div 
              className="form-group"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              transition={{ duration: 0.3 }}
            >
              <label className="form-label">Full Name</label>
              <div className="input-container">
                <User className="input-icon" />
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className={`form-input ${errors.name ? 'error' : ''}`}
                  placeholder="Enter your full name"
                />
              </div>
              {errors.name && <span className="error-message">{errors.name}</span>}
            </motion.div>
          )}

          <div className="form-group">
            <label className="form-label">Email Address</label>
            <div className="input-container">
              <Mail className="input-icon" />
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                className={`form-input ${errors.email ? 'error' : ''}`}
                placeholder="Enter your email"
              />
            </div>
            {errors.email && <span className="error-message">{errors.email}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <div className="input-container">
              <Lock className="input-icon" />
              <input
                type={showPassword ? 'text' : 'password'}
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                className={`form-input ${errors.password ? 'error' : ''}`}
                placeholder="Enter your password"
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff className="toggle-icon" /> : <Eye className="toggle-icon" />}
              </button>
            </div>
            {errors.password && <span className="error-message">{errors.password}</span>}
          </div>

          {!isLogin && (
            <motion.div 
              className="form-group"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              transition={{ duration: 0.3 }}
            >
              <label className="form-label">Confirm Password</label>
              <div className="input-container">
                <Lock className="input-icon" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  className={`form-input ${errors.confirmPassword ? 'error' : ''}`}
                  placeholder="Confirm your password"
                />
              </div>
              {errors.confirmPassword && <span className="error-message">{errors.confirmPassword}</span>}
            </motion.div>
          )}

          {errors.general && (
            <div className="error-message general-error">
              {errors.general}
            </div>
          )}

          <motion.button
            type="submit"
            className="submit-btn"
            disabled={loading}
            whileHover={{ scale: loading ? 1 : 1.02 }}
            whileTap={{ scale: loading ? 1 : 0.98 }}
          >
            {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Create Account')}
          </motion.button>
        </form>

        <div className="auth-footer">
          <p className="toggle-text">
            {isLogin ? "Don't have an account?" : "Already have an account?"}
          </p>
          <button 
            type="button"
            className="toggle-btn"
            onClick={toggleMode}
          >
            {isLogin ? 'Sign Up' : 'Sign In'}
          </button>
        </div>

        <div className="user-type-indicator">
          <span className={`type-badge type-${userType}`}>
            {userType === 'moderator' ? 'Moderator Access' : 'User Access'}
          </span>
        </div>
      </div>
    </motion.div>
  )
}

export default AuthForm

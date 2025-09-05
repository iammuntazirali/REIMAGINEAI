import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { gsap } from 'gsap'
import { 
  BarChart3, 
  Users, 
  Settings, 
  Bell, 
  Search, 
  Plus,
  TrendingUp,
  Activity,
  Zap,
  Shield
} from 'lucide-react'
import { dashboardAPI } from '../../services/api'
import './Dashboard.css'

const Dashboard = ({ userType = 'user' }) => {
  const dashboardRef = useRef(null)
  const cardsRef = useRef([])
  const [stats, setStats] = useState(null)
  const [activity, setActivity] = useState([])
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // GSAP animations for dashboard entrance
    const tl = gsap.timeline()
    
    tl.fromTo(dashboardRef.current, 
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }
    )
    
    tl.fromTo(cardsRef.current,
      { opacity: 0, y: 30, scale: 0.95 },
      { opacity: 1, y: 0, scale: 1, duration: 0.6, stagger: 0.1, ease: "back.out(1.7)" },
      "-=0.4"
    )

    // Fetch dashboard data
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      
      const [statsResponse, activityResponse, modelsResponse] = await Promise.all([
        dashboardAPI.getStats(),
        dashboardAPI.getActivity(),
        dashboardAPI.getModels()
      ])

      if (statsResponse.success) {
        setStats(statsResponse.data.stats)
      }

      if (activityResponse.success) {
        setActivity(activityResponse.data.activities)
      }

      if (modelsResponse.success) {
        setModels(modelsResponse.data.models)
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const statsData = [
    { title: 'Total Projects', value: stats?.totalProjects || '0', icon: BarChart3, color: 'blue', change: '+12%' },
    { title: 'Active Users', value: stats?.activeUsers || '0', icon: Users, color: 'green', change: '+8%' },
    { title: 'AI Models', value: stats?.aiModels || '0', icon: Zap, color: 'purple', change: '+3%' },
    { title: 'Security Score', value: stats?.securityScore || '0', icon: Shield, color: 'orange', change: '+2%' }
  ]

  return (
    <div ref={dashboardRef} className="dashboard-container">
      {/* Header */}
      <motion.div 
        className="dashboard-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="header-left">
          <h1 className="dashboard-title">
            Welcome back, {userType === 'moderator' ? 'Moderator' : 'User'}
          </h1>
          <p className="dashboard-subtitle">Here's what's happening with your AI projects</p>
        </div>
        <div className="header-right">
          <div className="search-container">
            <Search className="search-icon" />
            <input type="text" placeholder="Search..." className="search-input" />
          </div>
          <button className="notification-btn">
            <Bell className="notification-icon" />
            <span className="notification-badge">3</span>
          </button>
          <button className="create-btn">
            <Plus className="create-icon" />
            Create New
          </button>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <div className="stats-grid">
        {statsData.map((stat, index) => (
          <motion.div
            key={stat.title}
            ref={el => cardsRef.current[index] = el}
            className={`stat-card stat-card-${stat.color}`}
            whileHover={{ scale: 1.02, y: -5 }}
            transition={{ duration: 0.2 }}
          >
            <div className="stat-icon">
              <stat.icon className={`icon-${stat.color}`} />
            </div>
            <div className="stat-content">
              <h3 className="stat-value">{stat.value}</h3>
              <p className="stat-title">{stat.title}</p>
              <span className={`stat-change stat-change-${stat.color}`}>
                <TrendingUp className="change-icon" />
                {stat.change}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="content-grid">
        {/* AI Models Section */}
        <motion.div 
          className="content-card models-card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="card-header">
            <h3 className="card-title">
              <Zap className="card-icon" />
              AI Models
            </h3>
            <button className="card-action">View All</button>
          </div>
          <div className="models-list">
            {models.map((model, index) => (
              <motion.div
                key={model.id}
                className="model-item"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.3 + index * 0.1 }}
                whileHover={{ x: 5 }}
              >
                <div className="model-info">
                  <h4 className="model-name">{model.name}</h4>
                  <p className="model-status">{model.status}</p>
                </div>
                <div className="model-actions">
                  <button className="action-btn">Configure</button>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Activity Feed */}
        <motion.div 
          className="content-card activity-card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <div className="card-header">
            <h3 className="card-title">
              <Activity className="card-icon" />
              Recent Activity
            </h3>
            <button className="card-action">View All</button>
          </div>
          <div className="activity-list">
            {activity.map((activityItem, index) => (
              <motion.div
                key={activityItem.id}
                className={`activity-item activity-${activityItem.type}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.4 + index * 0.1 }}
              >
                <div className="activity-content">
                  <p className="activity-text">{activityItem.action}</p>
                  <span className="activity-time">{activityItem.time}</span>
                </div>
                <div className={`activity-indicator activity-${activityItem.type}`}></div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <motion.div 
        className="quick-actions"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
      >
        <h3 className="actions-title">Quick Actions</h3>
        <div className="actions-grid">
          <motion.button 
            className="action-card"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Plus className="action-icon" />
            <span>New Project</span>
          </motion.button>
          <motion.button 
            className="action-card"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <BarChart3 className="action-icon" />
            <span>Analytics</span>
          </motion.button>
          <motion.button 
            className="action-card"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Settings className="action-icon" />
            <span>Settings</span>
          </motion.button>
          <motion.button 
            className="action-card"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Users className="action-icon" />
            <span>Team</span>
          </motion.button>
        </div>
      </motion.div>
    </div>
  )
}

export default Dashboard

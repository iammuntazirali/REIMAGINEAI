import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { gsap } from 'gsap'
import { User, Shield, ArrowRight } from 'lucide-react'
import './UserTypeSelector.css'

const UserTypeSelector = ({ onSelectUserType }) => {
  const containerRef = useRef(null)
  const cardsRef = useRef([])

  useEffect(() => {
    // GSAP animations for entrance
    const tl = gsap.timeline()
    
    tl.fromTo(containerRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }
    )
    
    tl.fromTo(cardsRef.current,
      { opacity: 0, y: 40, scale: 0.9 },
      { opacity: 1, y: 0, scale: 1, duration: 0.6, stagger: 0.2, ease: "back.out(1.7)" },
      "-=0.4"
    )
  }, [])

  const userTypes = [
    {
      type: 'user',
      title: 'User',
      description: 'Access AI models, create projects, and collaborate with the community',
      icon: User,
      features: ['AI Model Access', 'Project Creation', 'Community Features', 'Basic Analytics'],
      color: 'blue',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    {
      type: 'moderator',
      title: 'Moderator',
      description: 'Manage users, monitor system health, and oversee AI model deployments',
      icon: Shield,
      features: ['User Management', 'System Monitoring', 'Model Deployment', 'Advanced Analytics'],
      color: 'purple',
      gradient: 'linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%)'
    }
  ]

  return (
    <div ref={containerRef} className="user-type-container">
      <motion.div 
        className="selector-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="selector-title">Choose Your Access Level</h1>
        <p className="selector-subtitle">
          Select the type of account that best fits your needs
        </p>
      </motion.div>

      <div className="user-types-grid">
        {userTypes.map((userType, index) => (
          <motion.div
            key={userType.type}
            ref={el => cardsRef.current[index] = el}
            className={`user-type-card user-type-${userType.color}`}
            whileHover={{ scale: 1.02, y: -10 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onSelectUserType(userType.type)}
          >
            <div className="card-header">
              <div 
                className="card-icon"
                style={{ background: userType.gradient }}
              >
                <userType.icon className="icon" />
              </div>
              <h3 className="card-title">{userType.title}</h3>
              <p className="card-description">{userType.description}</p>
            </div>

            <div className="card-features">
              <h4 className="features-title">What you get:</h4>
              <ul className="features-list">
                {userType.features.map((feature, featureIndex) => (
                  <motion.li
                    key={feature}
                    className="feature-item"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.6 + featureIndex * 0.1 }}
                  >
                    <div className="feature-dot" />
                    <span>{feature}</span>
                  </motion.li>
                ))}
              </ul>
            </div>

            <motion.button
              className="select-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span>Continue as {userType.title}</span>
              <ArrowRight className="arrow-icon" />
            </motion.button>
          </motion.div>
        ))}
      </div>

      <motion.div 
        className="selector-footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <p className="footer-text">
          You can change your access level later in your account settings
        </p>
      </motion.div>
    </div>
  )
}

export default UserTypeSelector

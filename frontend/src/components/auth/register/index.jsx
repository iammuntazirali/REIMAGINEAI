import React, { useState } from 'react'
import { Navigate, Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../../contexts/authContext'
import { doCreateUserWithEmailAndPassword } from '../../../firebase/auth'
import '../auth.css'

const Register = () => {

    const navigate = useNavigate()

    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [confirmPassword, setconfirmPassword] = useState('')
    const [isRegistering, setIsRegistering] = useState(false)
    const [errorMessage, setErrorMessage] = useState('')
    const [emailError, setEmailError] = useState('')
    const [passwordError, setPasswordError] = useState('')
    const [confirmPasswordError, setConfirmPasswordError] = useState('')
    const [passwordStrength, setPasswordStrength] = useState(0)

    const { userLoggedIn } = useAuth()

    const calculatePasswordStrength = (password) => {
        let strength = 0
        if (password.length >= 6) strength += 1
        if (password.length >= 8) strength += 1
        if (/[A-Z]/.test(password)) strength += 1
        if (/[a-z]/.test(password)) strength += 1
        if (/[0-9]/.test(password)) strength += 1
        if (/[^A-Za-z0-9]/.test(password)) strength += 1
        return strength
    }

    const handlePasswordChange = (e) => {
        const newPassword = e.target.value
        setPassword(newPassword)
        setPasswordStrength(calculatePasswordStrength(newPassword))
        
        // Clear password error when user starts typing
        if (passwordError) {
            setPasswordError('')
        }
        
        // Check password match in real-time
        if (confirmPassword && newPassword !== confirmPassword) {
            setConfirmPasswordError('Passwords do not match')
        } else if (confirmPassword && newPassword === confirmPassword) {
            setConfirmPasswordError('')
        }
    }

    const handleConfirmPasswordChange = (e) => {
        const newConfirmPassword = e.target.value
        setconfirmPassword(newConfirmPassword)
        
        if (newConfirmPassword && password !== newConfirmPassword) {
            setConfirmPasswordError('Passwords do not match')
        } else {
            setConfirmPasswordError('')
        }
    }

    const validateForm = () => {
        let isValid = true
        setEmailError('')
        setPasswordError('')
        setConfirmPasswordError('')
        setErrorMessage('')

        if (!email) {
            setEmailError('Email is required')
            isValid = false
        } else if (!/\S+@\S+\.\S+/.test(email)) {
            setEmailError('Please enter a valid email address')
            isValid = false
        }

        if (!password) {
            setPasswordError('Password is required')
            isValid = false
        } else if (password.length < 6) {
            setPasswordError('Password must be at least 6 characters')
            isValid = false
        }

        if (!confirmPassword) {
            setConfirmPasswordError('Please confirm your password')
            isValid = false
        } else if (password !== confirmPassword) {
            setConfirmPasswordError('Passwords do not match')
            isValid = false
        }

        return isValid
    }

    const onSubmit = async (e) => {
        e.preventDefault()
        if (!isRegistering && validateForm()) {
            setIsRegistering(true)
            try {
                await doCreateUserWithEmailAndPassword(email, password)
            } catch (error) {
                setErrorMessage(error.message || 'Failed to create account. Please try again.')
                setIsRegistering(false)
            }
        }
    }

    return (
        <>
            {userLoggedIn && (<Navigate to={'/home'} replace={true} />)}

            <main className="auth-container">
                <div className="auth-card">
                    <div className="auth-header">
                        <h1 className="auth-title">Create Account</h1>
                        <p className="auth-subtitle">Join us today and get started</p>
                    </div>

                    <form onSubmit={onSubmit} className="auth-form">
                        <div className="form-group">
                            <label htmlFor="email" className="form-label">
                                Email Address
                            </label>
                            <input
                                id="email"
                                type="email"
                                autoComplete="email"
                                required
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className={`form-input ${emailError ? 'border-red-500' : ''}`}
                                placeholder="Enter your email"
                                aria-describedby={emailError ? "email-error" : undefined}
                            />
                            {emailError && (
                                <div id="email-error" className="error-message">
                                    {emailError}
                                </div>
                            )}
                        </div>

                        <div className="form-group">
                            <label htmlFor="password" className="form-label">
                                Password
                            </label>
                            <input
                                id="password"
                                type="password"
                                autoComplete="new-password"
                                required
                                value={password}
                                onChange={handlePasswordChange}
                                className={`form-input ${passwordError ? 'border-red-500' : ''} ${password && !passwordError && passwordStrength >= 3 ? 'password-match' : ''}`}
                                placeholder="Create a password"
                                aria-describedby={passwordError ? "password-error" : undefined}
                            />
                            {password && (
                                <div className="password-strength">
                                    <div className="strength-bar">
                                        <div 
                                            className={`strength-fill strength-${Math.min(passwordStrength, 3)}`}
                                        ></div>
                                    </div>
                                    <span className={`strength-text ${
                                        passwordStrength === 1 ? 'weak' :
                                        passwordStrength === 2 ? 'fair' :
                                        passwordStrength >= 3 ? 'good' : ''
                                    }`}>
                                        {passwordStrength === 0 && 'Enter a password'}
                                        {passwordStrength === 1 && 'Weak'}
                                        {passwordStrength === 2 && 'Fair'}
                                        {passwordStrength === 3 && 'Good'}
                                        {passwordStrength >= 4 && 'Strong'}
                                    </span>
                                </div>
                            )}
                            {passwordError && (
                                <div id="password-error" className="error-message">
                                    {passwordError}
                                </div>
                            )}
                        </div>

                        <div className="form-group">
                            <label htmlFor="confirmPassword" className="form-label">
                                Confirm Password
                            </label>
                            <input
                                id="confirmPassword"
                                type="password"
                                autoComplete="off"
                                required
                                value={confirmPassword}
                                onChange={handleConfirmPasswordChange}
                                className={`form-input ${confirmPasswordError ? 'border-red-500' : ''} ${confirmPassword && !confirmPasswordError && password === confirmPassword ? 'password-match' : ''}`}
                                placeholder="Confirm your password"
                                aria-describedby={confirmPasswordError ? "confirm-password-error" : undefined}
                            />
                            {confirmPasswordError && (
                                <div id="confirm-password-error" className="error-message">
                                    {confirmPasswordError}
                                </div>
                            )}
                        </div>

                        {errorMessage && (
                            <div className="error-message">
                                {errorMessage}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={isRegistering}
                            className="auth-button"
                            aria-label={isRegistering ? "Creating account..." : "Create account"}
                        >
                            {isRegistering ? (
                                <>
                                    <span className="loading-spinner"></span>
                                    Creating Account...
                                </>
                            ) : (
                                'Create Account'
                            )}
                        </button>
                    </form>

                    <div className="auth-link">
                        <p>
                            Already have an account?{' '}
                            <Link to="/login">Sign in</Link>
                        </p>
                    </div>
                </div>
            </main>
        </>
    )
}

export default Register
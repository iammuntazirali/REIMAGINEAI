import React, { useState } from 'react'
import { Navigate, Link } from 'react-router-dom'
import { doSignInWithEmailAndPassword, doSignInWithGoogle } from '../../../firebase/auth'
import { useAuth } from '../../../contexts/authContext'
import '../auth.css'

const Login = () => {
    const { userLoggedIn } = useAuth()

    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [isSigningIn, setIsSigningIn] = useState(false)
    const [errorMessage, setErrorMessage] = useState('')
    const [emailError, setEmailError] = useState('')
    const [passwordError, setPasswordError] = useState('')

    const handleEmailChange = (e) => {
        const newEmail = e.target.value
        setEmail(newEmail)
        
        // Clear email error when user starts typing
        if (emailError) {
            setEmailError('')
        }
    }

    const handlePasswordChange = (e) => {
        const newPassword = e.target.value
        setPassword(newPassword)
        
        // Clear password error when user starts typing
        if (passwordError) {
            setPasswordError('')
        }
    }

    const validateForm = () => {
        let isValid = true
        setEmailError('')
        setPasswordError('')
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

        return isValid
    }

    const onSubmit = async (e) => {
        e.preventDefault()
        if (!isSigningIn && validateForm()) {
            setIsSigningIn(true)
            try {
                await doSignInWithEmailAndPassword(email, password)
            } catch (error) {
                setErrorMessage(error.message || 'Failed to sign in. Please try again.')
                setIsSigningIn(false)
            }
        }
    }

    const onGoogleSignIn = (e) => {
        e.preventDefault()
        if (!isSigningIn) {
            setIsSigningIn(true)
            doSignInWithGoogle().catch(err => {
                setErrorMessage(err.message || 'Failed to sign in with Google. Please try again.')
                setIsSigningIn(false)
            })
        }
    }

    return (
        <div>
            {userLoggedIn && (<Navigate to={'/home'} replace={true} />)}

            <main className="auth-container">
                <div className="auth-card">
                    <div className="auth-header">
                        <h1 className="auth-title">Welcome Back</h1>
                        <p className="auth-subtitle">Sign in to your account to continue</p>
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
                                onChange={handleEmailChange}
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
                                autoComplete="current-password"
                                required
                                value={password}
                                onChange={handlePasswordChange}
                                className={`form-input ${passwordError ? 'border-red-500' : ''}`}
                                placeholder="Enter your password"
                                aria-describedby={passwordError ? "password-error" : undefined}
                            />
                            {passwordError && (
                                <div id="password-error" className="error-message">
                                    {passwordError}
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
                            disabled={isSigningIn}
                            className="auth-button"
                            aria-label={isSigningIn ? "Signing in..." : "Sign in"}
                        >
                            {isSigningIn ? (
                                <>
                                    <span className="loading-spinner"></span>
                                    Signing In...
                                </>
                            ) : (
                                'Sign In'
                            )}
                        </button>
                    </form>

                    <div className="auth-divider">
                        <span>OR</span>
                    </div>

                    <button
                        disabled={isSigningIn}
                        onClick={onGoogleSignIn}
                        className="google-button"
                        aria-label={isSigningIn ? "Signing in with Google..." : "Sign in with Google"}
                    >
                        <svg className="google-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                        </svg>
                        {isSigningIn ? 'Signing In...' : 'Continue with Google'}
                    </button>

                    <div className="auth-link">
                        <p>
                            Don't have an account?{' '}
                            <Link to="/register">Sign up</Link>
                        </p>
                    </div>
                </div>
            </main>
        </div>
    )
}

export default Login
import React from 'react'
import { useAuth } from '../../contexts/authContext'
import { doSignOut } from '../../firebase/auth'
import { useNavigate } from 'react-router-dom'

const Home = () => {
    const { currentUser } = useAuth()
    const navigate = useNavigate()

    const handleLogout = async () => {
        try {
            await doSignOut()
            navigate('/')
        } catch (error) {
            console.error('Logout error:', error)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
            {/* Dashboard Header */}
            <div className="bg-black/20 backdrop-blur-lg border-b border-white/10">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center py-4">
                        <div className="flex items-center space-x-4">
                            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                                <span className="text-white font-bold text-lg">RA</span>
                            </div>
                            <h1 className="text-2xl font-bold text-white">REIMAGINE AI Dashboard</h1>
                        </div>
                        <div className="flex items-center space-x-4">
                            <span className="text-white/80">Welcome, {currentUser.displayName || currentUser.email}</span>
                            <button 
                                onClick={handleLogout}
                                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                            >
                                Logout
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Dashboard Content */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {/* AI Studio Card */}
                    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                                <span className="text-2xl">🧠</span>
                            </div>
                            <h3 className="text-xl font-semibold text-white">AI Studio</h3>
                        </div>
                        <p className="text-white/70 mb-4">Create amazing AI-generated content with our advanced models.</p>
                        <button className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-2 px-4 rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all">
                            Start Creating
                        </button>
                    </div>

                    {/* My Projects Card */}
                    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                                <span className="text-2xl">📁</span>
                            </div>
                            <h3 className="text-xl font-semibold text-white">My Projects</h3>
                        </div>
                        <p className="text-white/70 mb-4">View and manage all your AI creations in one place.</p>
                        <button className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-2 px-4 rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all">
                            View Projects
                        </button>
                    </div>

                    {/* Settings Card */}
                    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                        <div className="flex items-center space-x-3 mb-4">
                            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center">
                                <span className="text-2xl">⚙️</span>
                            </div>
                            <h3 className="text-xl font-semibold text-white">Settings</h3>
                        </div>
                        <p className="text-white/70 mb-4">Customize your AI experience and preferences.</p>
                        <button className="w-full bg-gradient-to-r from-green-500 to-emerald-500 text-white py-2 px-4 rounded-lg hover:from-green-600 hover:to-emerald-600 transition-all">
                            Open Settings
                        </button>
                    </div>
                </div>

                {/* Stats Section */}
                <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
                        <div className="text-2xl font-bold text-blue-400">12</div>
                        <div className="text-white/70 text-sm">Projects Created</div>
                    </div>
                    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
                        <div className="text-2xl font-bold text-purple-400">1.2K</div>
                        <div className="text-white/70 text-sm">AI Generations</div>
                    </div>
                    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
                        <div className="text-2xl font-bold text-green-400">24h</div>
                        <div className="text-white/70 text-sm">Processing Time</div>
                    </div>
                    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
                        <div className="text-2xl font-bold text-orange-400">98%</div>
                        <div className="text-white/70 text-sm">Success Rate</div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Home
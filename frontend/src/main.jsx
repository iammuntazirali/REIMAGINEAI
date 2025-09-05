import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
<<<<<<< HEAD
import { AuthProvider } from './contexts/authContext'
=======
>>>>>>> 28480848ffb3f415cfb3ea3b9901bb587ccbb261
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
<<<<<<< HEAD
      <AuthProvider>
        <App />
      </AuthProvider>
=======
      <App />
>>>>>>> 28480848ffb3f415cfb3ea3b9901bb587ccbb261
    </BrowserRouter>
  </StrictMode>,
)


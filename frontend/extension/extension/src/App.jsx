import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div className="flex justify-center items-center gap-4 p-8">
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo hover:animate-spin" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react animate-spin" alt="React logo" />
        </a>
      </div>
      <h1 className="text-4xl font-bold text-center mb-8 text-purple-600">Extension + React</h1>
      <div className="card text-center">
        <button 
          className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded transition-colors duration-200"
          onClick={() => setCount((count) => count + 1)}
        >
          count is {count}
        </button>
        <p className="mt-4 text-gray-600">
          Edit <code className="bg-gray-200 px-2 py-1 rounded">src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs text-center text-sm text-gray-500 mt-8">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App

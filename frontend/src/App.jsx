import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import IndexPage from './pages/IndexPage'
import VideosPage from './pages/VideosPage'
import VideoDetailPage from './pages/VideoDetailPage'
import TaskDetailPage from './pages/TaskDetailPage'

function Nav() {
  return (
    <nav className="bg-gray-900 border-b border-gray-800 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Link to="/" className="text-xl font-bold text-blue-400 hover:text-blue-300">
          ForgeIndex
        </Link>
        <div className="flex gap-6">
          <Link to="/" className="text-gray-300 hover:text-white transition-colors">
            Dashboard
          </Link>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <main className="max-w-7xl mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<IndexPage />} />
          <Route path="/videos" element={<VideosPage />} />
          <Route path="/videos/:id" element={<VideoDetailPage />} />
          <Route path="/tasks/:id" element={<TaskDetailPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}

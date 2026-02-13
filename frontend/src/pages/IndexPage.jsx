import { useState, useEffect, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { getStats, createIngestion, listTasks, getTask } from '../api'

function StatCard({ label, value, color = 'text-white' }) {
  return (
    <div className="card">
      <p className="text-gray-400 text-sm mb-1">{label}</p>
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
    </div>
  )
}

export default function IndexPage() {
  const [stats, setStats] = useState(null)
  const [tasks, setTasks] = useState([])
  const [query, setQuery] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [message, setMessage] = useState('')
  const [messageType, setMessageType] = useState('info')
  // Real-time thinking state
  const [activeTaskId, setActiveTaskId] = useState(null)
  const [thinkingSteps, setThinkingSteps] = useState([])
  const pollRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 5000)
    return () => {
      clearInterval(interval)
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  async function loadData() {
    try {
      const [s, t] = await Promise.all([getStats(), listTasks({ limit: 10 })])
      setStats(s)
      setTasks(t)
    } catch (e) {
      console.error('Failed to load data:', e)
    }
  }

  function startPolling(taskId) {
    setActiveTaskId(taskId)
    setThinkingSteps([{ text: 'Task created, starting...', time: Date.now() }])

    pollRef.current = setInterval(async () => {
      try {
        const task = await getTask(taskId)
        if (task.current_step) {
          setThinkingSteps((prev) => {
            const last = prev[prev.length - 1]
            if (last && last.text === task.current_step) return prev
            return [...prev, { text: task.current_step, time: Date.now() }]
          })
        }
        if (task.status === 'completed' || task.status === 'failed') {
          clearInterval(pollRef.current)
          pollRef.current = null
          setActiveTaskId(null)
          loadData()
        }
      } catch (e) {
        // ignore poll errors
      }
    }, 1500)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim()) return
    setSubmitting(true)
    setMessage('')
    setMessageType('info')
    setThinkingSteps([])
    try {
      const result = await createIngestion(query)
      setMessage(`Task #${result.task_id} created: ${result.message}`)
      setMessageType('success')
      setQuery('')
      loadData()
      // Start polling for real-time updates
      startPolling(result.task_id)
    } catch (e) {
      if (e.code === 'gdpr_query_blocked') {
        setMessage(e.message)
        setMessageType('gdpr')
      } else {
        setMessage(e.message || String(e))
        setMessageType('error')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Dashboard</h1>

      {/* Stats — only fully processed videos */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Videos in Library" value={stats?.total_videos ?? '—'} color="text-blue-400" />
        <StatCard label="Processing" value={stats?.processing_videos ?? '—'} color="text-yellow-400" />
        <StatCard label="Failed" value={stats?.failed_videos ?? '—'} color="text-red-400" />
        <StatCard label="GDPR Blocked" value={stats?.gdpr_blocked_videos ?? '—'} color="text-orange-400" />
      </div>

      {/* Ingestion Form */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Ingest Videos</h2>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex gap-4">
            <input
              className="input flex-1"
              placeholder="e.g., 20 cooking videos from instagram"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <button className="btn-primary w-fit" disabled={submitting}>
            {submitting ? 'Submitting...' : 'Start Ingestion'}
          </button>

          {/* GDPR / error messages */}
          {message && !activeTaskId && (
            <div className={`text-sm rounded-lg p-3 ${
              messageType === 'gdpr'
                ? 'bg-red-900/20 border border-red-800 text-red-300'
                : messageType === 'error'
                ? 'bg-red-900/20 text-red-400'
                : messageType === 'success'
                ? 'bg-green-900/20 text-green-400'
                : 'text-gray-400'
            }`}>
              {messageType === 'gdpr' && <span className="font-medium block mb-1">GDPR Compliance Check Failed</span>}
              {message}
            </div>
          )}

          {/* Real-time thinking steps */}
          {thinkingSteps.length > 0 && (
            <div className="space-y-1 pt-1">
              {thinkingSteps.map((step, i) => {
                const isLatest = i === thinkingSteps.length - 1
                const isActive = isLatest && activeTaskId
                return (
                  <div key={i} className="flex items-center gap-2">
                    {isActive ? (
                      <span className="flex gap-0.5">
                        <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '100ms' }} />
                        <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '200ms' }} />
                      </span>
                    ) : (
                      <span className="text-green-600 text-xs">&#10003;</span>
                    )}
                    <span className={`text-xs ${isActive ? 'text-gray-300' : 'text-gray-600'}`}>
                      {step.text}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </form>
      </div>

      {/* Recent Tasks */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Recent Tasks</h2>
        {tasks.length === 0 ? (
          <p className="text-gray-500">No tasks yet</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-400 border-b border-gray-800">
                  <th className="pb-2 pr-4">ID</th>
                  <th className="pb-2 pr-4">Query</th>
                  <th className="pb-2 pr-4">Platform</th>
                  <th className="pb-2 pr-4">Status</th>
                  <th className="pb-2 pr-4">Progress</th>
                  <th className="pb-2">Created</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr key={t.id} className="border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors cursor-pointer"
                      onClick={() => navigate(`/tasks/${t.id}`)}>
                    <td className="py-2 pr-4 text-gray-300">
                      <Link to={`/tasks/${t.id}`} className="text-blue-400 hover:text-blue-300">#{t.id}</Link>
                    </td>
                    <td className="py-2 pr-4">
                      <div>{t.query}</div>
                      {t.status === 'running' && t.current_step && (
                        <div className="flex items-center gap-1.5 mt-0.5">
                          <span className="flex gap-0.5">
                            <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '100ms' }} />
                            <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '200ms' }} />
                          </span>
                          <span className="text-xs text-gray-500">{t.current_step}</span>
                        </div>
                      )}
                    </td>
                    <td className="py-2 pr-4 text-gray-400">{t.platform}</td>
                    <td className="py-2 pr-4">
                      <StatusBadge status={t.status} />
                    </td>
                    <td className="py-2 pr-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-gray-800 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full transition-all"
                            style={{ width: `${(t.progress * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <span className="text-gray-500 text-xs">
                          {t.processed_videos || 0}/{t.total_videos || 0}
                        </span>
                      </div>
                    </td>
                    <td className="py-2 text-gray-500 text-xs">
                      {new Date(t.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function StatusBadge({ status }) {
  const colors = {
    pending: 'bg-gray-700 text-gray-300',
    running: 'bg-yellow-900 text-yellow-300',
    completed: 'bg-green-900 text-green-300',
    failed: 'bg-red-900 text-red-300',
  }
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[status] || colors.pending}`}>
      {status}
    </span>
  )
}

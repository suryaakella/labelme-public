import { useState, useEffect, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getTaskVideos } from '../api'

export default function TaskDetailPage() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [initialLoading, setInitialLoading] = useState(true)
  const [error, setError] = useState('')
  const pollRef = useRef(null)

  useEffect(() => {
    loadData(true)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [id])

  // Start/stop polling based on task status
  useEffect(() => {
    if (!data) return
    const status = data.task.status
    const shouldPoll = status === 'running' || status === 'pending'

    if (shouldPoll && !pollRef.current) {
      pollRef.current = setInterval(() => loadData(false), 2000)
    } else if (!shouldPoll && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [data?.task?.status])

  async function loadData(isInitial) {
    try {
      const result = await getTaskVideos(id)
      setData(result)
    } catch (e) {
      setError(e.message)
    } finally {
      if (isInitial) setInitialLoading(false)
    }
  }

  if (initialLoading) return <p className="text-gray-400">Loading...</p>
  if (error) return <p className="text-red-400">Error: {error}</p>
  if (!data) return <p className="text-gray-400">Task not found</p>

  const { task, videos } = data
  const isActive = task.status === 'running' || task.status === 'pending'

  return (
    <div className="space-y-8">
      <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm">
        &larr; Back to dashboard
      </Link>

      {/* Task Info */}
      <div className="card">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold mb-1">Task #{task.id}</h1>
            <p className="text-gray-300 text-lg">{task.query}</p>
          </div>
          <StatusBadge status={task.status} />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500 block">Platform</span>
            <span className="text-gray-200">{task.platform}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Max Results</span>
            <span className="text-gray-200">{task.max_results}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Progress</span>
            <div className="flex items-center gap-2 mt-1">
              <div className="w-24 bg-gray-800 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${(task.progress * 100).toFixed(0)}%` }}
                />
              </div>
              <span className="text-gray-400 text-xs">
                {task.processed_videos || 0}/{task.total_videos || 0}
              </span>
            </div>
          </div>
          <div>
            <span className="text-gray-500 block">Created</span>
            <span className="text-gray-200 text-xs">
              {new Date(task.created_at).toLocaleString()}
            </span>
          </div>
        </div>

        {/* Task-level current step with animated dots */}
        {task.current_step && isActive && (
          <div className="flex items-center gap-2 mt-3">
            <BouncingDots />
            <span className="text-gray-300 text-sm">{task.current_step}</span>
          </div>
        )}
        {task.current_step && !isActive && task.status === 'completed' && (
          <p className="text-green-400 text-sm mt-3">{task.current_step}</p>
        )}
        {task.error_message && task.status === 'failed' && (
          <p className="text-red-400 text-sm mt-3">{task.error_message}</p>
        )}
      </div>

      {/* Videos */}
      <div>
        <h2 className="text-xl font-semibold mb-4">
          Videos ({videos.length})
        </h2>

        {videos.length === 0 ? (
          <div className="card">
            <p className="text-gray-500">No videos associated with this task yet.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {videos.map((v) => (
              <VideoCard key={v.id} video={v} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function VideoCard({ video: v }) {
  const isInProgress = ['downloading', 'processing', 'labeling', 'pending'].includes(v.status)
  const isFailed = v.status === 'failed'
  const isGdprBlocked = v.status === 'gdpr_blocked'
  const isCompleted = v.status === 'completed'

  return (
    <Link
      to={`/videos/${v.id}`}
      className="card hover:border-gray-600 transition-colors block"
    >
      {/* Thumbnail */}
      {v.thumbnail_url ? (
        <img
          src={v.thumbnail_url}
          alt={v.title}
          className="w-full h-40 object-cover rounded-lg mb-3"
        />
      ) : (
        <div className="w-full h-40 bg-gray-800 rounded-lg mb-3 flex items-center justify-center text-gray-600 text-sm">
          No thumbnail
        </div>
      )}

      {/* Title */}
      <h3 className="font-medium text-gray-200 mb-1 line-clamp-2 text-sm">
        {v.title || 'Untitled'}
      </h3>

      {/* Meta row */}
      <div className="flex items-center gap-2 text-xs text-gray-400 mb-2 flex-wrap">
        <span className="bg-gray-800 px-1.5 py-0.5 rounded">{v.platform}</span>
        <VideoStatusBadge status={v.status} />
        {v.duration_sec && (
          <span>
            {Math.floor(v.duration_sec / 60)}:{String(Math.floor(v.duration_sec % 60)).padStart(2, '0')}
          </span>
        )}
      </div>

      {/* In-progress: show pipeline step */}
      {isInProgress && v.current_step && (
        <div className="flex items-center gap-2 mb-2">
          <BouncingDots />
          <span className="text-xs text-gray-400">{v.current_step}</span>
        </div>
      )}
      {isInProgress && !v.current_step && (
        <div className="flex items-center gap-2 mb-2">
          <BouncingDots />
          <span className="text-xs text-gray-400">Waiting...</span>
        </div>
      )}

      {/* Failed */}
      {isFailed && v.error_message && (
        <p className="text-red-400 text-xs mb-2 line-clamp-2">{v.error_message}</p>
      )}

      {/* GDPR blocked */}
      {isGdprBlocked && (
        <div className="bg-red-900/20 border border-red-800/50 rounded px-2 py-1 mb-2">
          <span className="text-red-300 text-xs">
            GDPR {v.gdpr_status ? `- ${v.gdpr_status}` : 'Blocked'}
          </span>
        </div>
      )}

      {/* Completed: show intelligence summary */}
      {isCompleted && (
        <div className="space-y-2 mt-1">
          {/* Tier badges row */}
          <div className="flex flex-wrap gap-1.5">
            {v.performance_tier && (
              <TierBadge label={v.performance_tier} type="performance" />
            )}
            {v.brand_safety_tier && (
              <TierBadge label={v.brand_safety_tier} type="brand_safety" />
            )}
            {v.sentiment_label && (
              <SentimentBadge label={v.sentiment_label} avg={v.sentiment_avg} />
            )}
            {v.is_ai_generated && (
              <span className="bg-purple-900/60 text-purple-300 text-xs px-1.5 py-0.5 rounded">
                AI Generated
              </span>
            )}
          </div>

          {/* Content categories */}
          {v.content_categories && v.content_categories.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {v.content_categories.map((cat) => (
                <span key={cat} className="bg-gray-800 text-gray-300 text-xs px-1.5 py-0.5 rounded">
                  {cat}
                </span>
              ))}
            </div>
          )}

          {/* Engagement stats */}
          {v.engagement && Object.keys(v.engagement).length > 0 && (
            <div className="flex gap-3 text-xs text-gray-500">
              {v.engagement.plays > 0 && <span>{formatCount(v.engagement.plays)} plays</span>}
              {v.engagement.likes > 0 && <span>{formatCount(v.engagement.likes)} likes</span>}
            </div>
          )}

          {/* Creator */}
          {v.creator_username && (
            <p className="text-xs text-gray-500">@{v.creator_username}</p>
          )}

          {/* Tags */}
          {v.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {v.tags.slice(0, 4).map((tag) => (
                <span key={tag} className="bg-blue-900/50 text-blue-300 text-xs px-1.5 py-0.5 rounded">
                  {tag}
                </span>
              ))}
              {v.tags.length > 4 && (
                <span className="text-gray-500 text-xs">+{v.tags.length - 4}</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Non-completed, non-in-progress: still show basic info */}
      {!isCompleted && !isInProgress && !isFailed && !isGdprBlocked && (
        <>
          {v.creator_username && (
            <p className="text-xs text-gray-500 mb-1">@{v.creator_username}</p>
          )}
          {v.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {v.tags.slice(0, 4).map((tag) => (
                <span key={tag} className="bg-blue-900/50 text-blue-300 text-xs px-1.5 py-0.5 rounded">
                  {tag}
                </span>
              ))}
              {v.tags.length > 4 && (
                <span className="text-gray-500 text-xs">+{v.tags.length - 4}</span>
              )}
            </div>
          )}
        </>
      )}
    </Link>
  )
}

function TierBadge({ label, type }) {
  const colors = {
    performance: {
      viral: 'bg-yellow-900/60 text-yellow-300',
      excellent: 'bg-green-900/60 text-green-300',
      good: 'bg-blue-900/60 text-blue-300',
      average: 'bg-gray-700 text-gray-300',
      below_average: 'bg-red-900/40 text-red-300',
    },
    brand_safety: {
      safe: 'bg-green-900/60 text-green-300',
      low_risk: 'bg-blue-900/60 text-blue-300',
      medium_risk: 'bg-yellow-900/60 text-yellow-300',
      high_risk: 'bg-red-900/60 text-red-300',
    },
  }
  const prefix = type === 'performance' ? '' : 'Safety: '
  const colorMap = colors[type] || {}
  const cls = colorMap[label] || 'bg-gray-700 text-gray-300'

  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>
      {prefix}{label.replace(/_/g, ' ')}
    </span>
  )
}

function SentimentBadge({ label, avg }) {
  const colors = {
    positive: 'bg-green-900/60 text-green-300',
    neutral: 'bg-gray-700 text-gray-300',
    negative: 'bg-red-900/60 text-red-300',
  }
  const cls = colors[label] || colors.neutral
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>
      {label}{avg != null ? ` (${avg.toFixed(2)})` : ''}
    </span>
  )
}

function BouncingDots() {
  return (
    <span className="flex gap-0.5">
      <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
      <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '100ms' }} />
      <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '200ms' }} />
    </span>
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

function VideoStatusBadge({ status }) {
  const colors = {
    pending: 'bg-gray-700 text-gray-300',
    downloading: 'bg-blue-900 text-blue-300',
    processing: 'bg-yellow-900 text-yellow-300',
    labeling: 'bg-purple-900 text-purple-300',
    completed: 'bg-green-900 text-green-300',
    failed: 'bg-red-900 text-red-300',
    gdpr_blocked: 'bg-red-900 text-red-300',
  }
  const labels = { gdpr_blocked: 'GDPR Blocked' }
  return (
    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${colors[status] || colors.pending}`}>
      {labels[status] || status}
    </span>
  )
}

function formatCount(n) {
  if (n == null) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return String(n)
}

import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { listVideos } from '../api'

export default function VideosPage() {
  const [data, setData] = useState(null)
  const [page, setPage] = useState(1)
  const [platform, setPlatform] = useState('')
  const [status, setStatus] = useState('completed')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadVideos()
  }, [page, platform, status])

  async function loadVideos() {
    setLoading(true)
    try {
      const res = await listVideos({
        page,
        pageSize: 20,
        platform: platform || undefined,
        status: status || undefined,
      })
      setData(res)
    } catch (e) {
      console.error('Failed to load videos:', e)
    } finally {
      setLoading(false)
    }
  }

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Videos</h1>
        {data && <span className="text-gray-400 text-sm">{data.total} total</span>}
      </div>

      {/* Filters */}
      <div className="flex gap-4 items-center">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Platform</label>
          <select className="input" value={platform} onChange={(e) => { setPlatform(e.target.value); setPage(1) }}>
            <option value="">All</option>
            <option value="youtube">YouTube</option>
            <option value="instagram">Instagram</option>
            <option value="tiktok">TikTok</option>
          </select>
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Status</label>
          <select className="input" value={status} onChange={(e) => { setStatus(e.target.value); setPage(1) }}>
            <option value="">All</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
            <option value="downloading">Downloading</option>
            <option value="failed">Failed</option>
            <option value="gdpr_blocked">GDPR Blocked</option>
          </select>
        </div>
      </div>

      {loading ? (
        <p className="text-gray-400">Loading...</p>
      ) : !data?.videos?.length ? (
        <p className="text-gray-500">No videos found</p>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {data.videos.map((v) => (
              <Link
                key={v.id}
                to={`/videos/${v.id}`}
                className="card hover:border-blue-600 transition-colors group"
              >
                {v.thumbnail_url ? (
                  <img
                    src={v.thumbnail_url}
                    alt={v.title}
                    className="w-full h-40 object-cover rounded-lg mb-3"
                  />
                ) : (
                  <div className="w-full h-40 bg-gray-800 rounded-lg mb-3 flex items-center justify-center text-gray-600">
                    No thumbnail
                  </div>
                )}
                <h3 className="font-medium text-gray-100 group-hover:text-blue-400 transition-colors line-clamp-2">
                  {v.title || 'Untitled'}
                </h3>
                {v.creator_username && (
                  <p className="text-gray-500 text-xs mt-1">@{v.creator_username}</p>
                )}
                <div className="flex items-center justify-between mt-2 text-sm">
                  <span className="text-gray-400">{v.platform}</span>
                  <StatusBadge status={v.status} />
                </div>
                <div className="flex items-center justify-between mt-1 text-xs text-gray-500">
                  {v.duration_sec ? (
                    <span>{Math.floor(v.duration_sec / 60)}:{String(Math.floor(v.duration_sec % 60)).padStart(2, '0')}</span>
                  ) : <span />}
                  <span>{new Date(v.created_at).toLocaleDateString()}</span>
                </div>
                {v.engagement && (v.engagement.plays > 0 || v.engagement.likes > 0) && (
                  <div className="flex gap-3 mt-1 text-xs text-gray-500">
                    {v.engagement.plays > 0 && <span>{formatCount(v.engagement.plays)} plays</span>}
                    {v.engagement.likes > 0 && <span>{formatCount(v.engagement.likes)} likes</span>}
                    {v.engagement.comments > 0 && <span>{formatCount(v.engagement.comments)} comments</span>}
                  </div>
                )}
                {v.analytics_summary && (
                  <div className="flex gap-2 mt-2">
                    <PerformanceBadge tier={v.analytics_summary.performance_tier} />
                    <BrandSafetyBadge tier={v.analytics_summary.brand_safety_tier} />
                  </div>
                )}
                {v.gdpr_status && (
                  <GDPRBadge status={v.gdpr_flags?.status} message={v.gdpr_status} />
                )}
                {v.tags?.length > 0 && (
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
              </Link>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-3">
              <button
                className="btn-secondary text-sm"
                disabled={page <= 1}
                onClick={() => setPage(page - 1)}
              >
                Previous
              </button>
              <span className="text-gray-400 text-sm">
                Page {page} of {totalPages}
              </span>
              <button
                className="btn-secondary text-sm"
                disabled={page >= totalPages}
                onClick={() => setPage(page + 1)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function StatusBadge({ status }) {
  const colors = {
    pending: 'bg-gray-700 text-gray-300',
    downloading: 'bg-blue-900 text-blue-300',
    processing: 'bg-yellow-900 text-yellow-300',
    labeling: 'bg-purple-900 text-purple-300',
    completed: 'bg-green-900 text-green-300',
    failed: 'bg-red-900 text-red-300',
    gdpr_blocked: 'bg-red-900 text-red-300',
  }
  const labels = {
    gdpr_blocked: 'GDPR Blocked',
  }
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[status] || colors.pending}`}>
      {labels[status] || status}
    </span>
  )
}

function PerformanceBadge({ tier }) {
  if (!tier) return null
  const colors = {
    viral: 'bg-green-900/50 text-green-300',
    excellent: 'bg-blue-900/50 text-blue-300',
    good: 'bg-teal-900/50 text-teal-300',
    average: 'bg-gray-700 text-gray-400',
    below_average: 'bg-red-900/50 text-red-300',
  }
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${colors[tier] || colors.average}`}>
      {tier.replace('_', ' ')}
    </span>
  )
}

function BrandSafetyBadge({ tier }) {
  if (!tier) return null
  const colors = {
    safe: 'bg-green-900/50 text-green-300',
    low_risk: 'bg-yellow-900/50 text-yellow-300',
    medium_risk: 'bg-orange-900/50 text-orange-300',
    high_risk: 'bg-red-900/50 text-red-300',
  }
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${colors[tier] || colors.safe}`}>
      {tier.replace('_', ' ')}
    </span>
  )
}

function GDPRBadge({ status, message }) {
  if (!status) return null
  const colors = {
    clean: 'bg-green-900/50 text-green-300',
    blocked: 'bg-red-900/50 text-red-300',
    unverified: 'bg-yellow-900/50 text-yellow-300',
    error: 'bg-orange-900/50 text-orange-300',
  }
  const labels = {
    clean: 'GDPR OK',
    blocked: 'GDPR Blocked',
    unverified: 'GDPR Unverified',
    error: 'GDPR Error',
  }
  return (
    <div className="mt-2">
      <span className={`text-xs px-1.5 py-0.5 rounded ${colors[status] || colors.error}`} title={message}>
        {labels[status] || 'GDPR Unknown'}
      </span>
    </div>
  )
}

function formatCount(n) {
  if (n == null) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return String(n)
}

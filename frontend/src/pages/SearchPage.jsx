import { useState } from 'react'
import { Link } from 'react-router-dom'
import { searchVideos } from '../api'

const THINKING_STEPS = [
  'Encoding query with CLIP model...',
  'Searching vector embeddings...',
  'Ranking by cosine similarity...',
  'Filtering results...',
  'Loading video metadata...',
]

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [platform, setPlatform] = useState('')
  const [minSimilarity, setMinSimilarity] = useState(0)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [thinkingStep, setThinkingStep] = useState(0)
  const [error, setError] = useState('')

  async function handleSearch(e) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError('')
    setThinkingStep(0)

    // Cycle through thinking steps while waiting
    const stepInterval = setInterval(() => {
      setThinkingStep((prev) => Math.min(prev + 1, THINKING_STEPS.length - 1))
    }, 600)

    try {
      const data = await searchVideos(query, {
        platform: platform || undefined,
        minSimilarity: minSimilarity || undefined,
      })
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      clearInterval(stepInterval)
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Semantic Search</h1>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="flex gap-4 items-end">
        <div className="flex-1">
          <input
            className="input w-full text-lg"
            placeholder="Describe what you're looking for..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <button className="btn-primary" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {/* Filters */}
      <div className="flex gap-4 items-center">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Platform</label>
          <select className="input" value={platform} onChange={(e) => setPlatform(e.target.value)}>
            <option value="">All</option>
            <option value="youtube">YouTube</option>
            <option value="instagram">Instagram</option>
            <option value="tiktok">TikTok</option>
          </select>
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Min Similarity</label>
          <input
            className="input w-24"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minSimilarity}
            onChange={(e) => setMinSimilarity(Number(e.target.value))}
          />
        </div>
      </div>

      {/* Thinking Indicator */}
      {loading && (
        <div className="flex items-start gap-3 py-3">
          <div className="flex gap-1 mt-1">
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          <div className="text-sm">
            {THINKING_STEPS.slice(0, thinkingStep + 1).map((step, i) => (
              <div
                key={i}
                className={`transition-opacity duration-300 ${
                  i === thinkingStep ? 'text-blue-400' : 'text-gray-600'
                }`}
              >
                {step}
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <p className="text-red-400">{error}</p>}

      {/* Results */}
      {results && (
        <div>
          <p className="text-gray-400 mb-4">
            {results.total} result{results.total !== 1 ? 's' : ''} for "{results.query}"
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.results.map((r) => (
              <Link
                key={r.video_id}
                to={`/videos/${r.video_id}`}
                className="card hover:border-blue-600 transition-colors group"
              >
                {r.thumbnail_url ? (
                  <img
                    src={r.thumbnail_url}
                    alt={r.title}
                    className="w-full h-40 object-cover rounded-lg mb-3"
                  />
                ) : (
                  <div className="w-full h-40 bg-gray-800 rounded-lg mb-3 flex items-center justify-center text-gray-600">
                    No thumbnail
                  </div>
                )}
                <h3 className="font-medium text-gray-100 group-hover:text-blue-400 transition-colors line-clamp-2">
                  {r.title || 'Untitled'}
                </h3>
                {r.creator_username && (
                  <p className="text-gray-500 text-xs mt-1">@{r.creator_username}</p>
                )}
                <div className="flex items-center justify-between mt-2 text-sm">
                  <span className="text-gray-400">{r.platform}</span>
                  <span className="text-blue-400 font-mono">
                    {(r.similarity * 100).toFixed(1)}% match
                  </span>
                </div>
                <div className="flex items-center justify-between mt-1 text-xs text-gray-500">
                  {r.duration_sec ? (
                    <span>{Math.floor(r.duration_sec / 60)}:{String(Math.floor(r.duration_sec % 60)).padStart(2, '0')}</span>
                  ) : <span />}
                  {r.engagement && (r.engagement.plays > 0 || r.engagement.likes > 0) && (
                    <span>
                      {r.engagement.plays > 0 && `${formatCount(r.engagement.plays)} plays`}
                      {r.engagement.plays > 0 && r.engagement.likes > 0 && ' · '}
                      {r.engagement.likes > 0 && `${formatCount(r.engagement.likes)} likes`}
                    </span>
                  )}
                </div>
                {r.analytics_summary && (
                  <div className="flex gap-2 mt-2">
                    <PerformanceBadge tier={r.analytics_summary.performance_tier} />
                    <BrandSafetyBadge tier={r.analytics_summary.brand_safety_tier} />
                  </div>
                )}
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
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

function formatCount(n) {
  if (n == null) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return String(n)
}

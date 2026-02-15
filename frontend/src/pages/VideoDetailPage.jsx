import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getVideo } from '../api'

export default function VideoDetailPage() {
  const { id } = useParams()
  const [video, setVideo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    loadVideo()
  }, [id])

  async function loadVideo() {
    setLoading(true)
    try {
      const data = await getVideo(id)
      setVideo(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <p className="text-gray-400">Loading...</p>
  if (error) return <p className="text-red-400">Error: {error}</p>
  if (!video) return <p className="text-gray-400">Video not found</p>

  const ann = video.annotation
  const isV2 = ann && ann.version === 2

  return (
    <div className="space-y-6">
      <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm">
        &larr; Back to dashboard
      </Link>

      {/* GDPR Status Banner */}
      {video.gdpr_status && <GDPRBanner gdprFlags={video.gdpr_flags} gdprStatus={video.gdpr_status} />}

      {/* Header: Video Player + Metadata */}
      <div className="flex flex-col md:flex-row gap-8">
        {/* Video Player / Thumbnail */}
        <div className="w-full md:w-96 flex-shrink-0">
          {video.storage_url ? (
            <video
              src={video.storage_url}
              controls
              className="w-full rounded-lg bg-black"
              poster={video.thumbnail_url}
            />
          ) : video.thumbnail_url ? (
            <img src={video.thumbnail_url} alt={video.title} className="w-full rounded-lg" />
          ) : (
            <div className="w-full h-56 bg-gray-800 rounded-lg flex items-center justify-center text-gray-600">
              No preview available
            </div>
          )}
        </div>

        <div className="flex-1 min-w-0">
          {/* Creator Info */}
          {video.creator_username && (
            <div className="flex items-center gap-3 mb-3">
              {video.creator_avatar_url ? (
                <img
                  src={video.creator_avatar_url}
                  alt={video.creator_username}
                  className="w-10 h-10 rounded-full object-cover"
                />
              ) : (
                <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center text-gray-400 text-sm font-bold">
                  {video.creator_username.charAt(0).toUpperCase()}
                </div>
              )}
              <div>
                <span className="font-medium text-gray-200">@{video.creator_username}</span>
                {video.creator_followers != null && video.creator_followers > 0 && (
                  <span className="text-gray-500 text-xs ml-2">
                    {formatCount(video.creator_followers)} followers
                  </span>
                )}
              </div>
            </div>
          )}

          <h1 className="text-2xl font-bold mb-2">{video.title || 'Untitled'}</h1>
          <div className="flex gap-3 items-center text-sm text-gray-400 mb-4 flex-wrap">
            <span className="bg-gray-800 px-2 py-0.5 rounded">{video.platform}</span>
            <StatusBadge status={video.status} />
            {video.duration_sec && (
              <span>
                {Math.floor(video.duration_sec / 60)}:{String(Math.floor(video.duration_sec % 60)).padStart(2, '0')}
              </span>
            )}
            {video.width && video.height && (
              <span>{video.width}x{video.height}</span>
            )}
            {video.posted_at && (
              <span>Posted {new Date(video.posted_at).toLocaleDateString()}</span>
            )}
          </div>

          {/* Engagement Bar */}
          {video.engagement && Object.keys(video.engagement).length > 0 && (
            <div className="flex flex-wrap gap-4 mb-4 text-sm">
              {video.engagement.plays > 0 && (
                <span className="text-gray-300">
                  <span className="text-gray-500 mr-1">Plays</span>
                  {formatCount(video.engagement.plays)}
                </span>
              )}
              {video.engagement.likes > 0 && (
                <span className="text-gray-300">
                  <span className="text-gray-500 mr-1">Likes</span>
                  {formatCount(video.engagement.likes)}
                </span>
              )}
              {video.engagement.comments > 0 && (
                <span className="text-gray-300">
                  <span className="text-gray-500 mr-1">Comments</span>
                  {formatCount(video.engagement.comments)}
                </span>
              )}
              {video.engagement.shares > 0 && (
                <span className="text-gray-300">
                  <span className="text-gray-500 mr-1">Shares</span>
                  {formatCount(video.engagement.shares)}
                </span>
              )}
              {video.engagement.saves > 0 && (
                <span className="text-gray-300">
                  <span className="text-gray-500 mr-1">Saves</span>
                  {formatCount(video.engagement.saves)}
                </span>
              )}
            </div>
          )}

          {/* Meta Badges */}
          <div className="flex flex-wrap gap-2 mb-4">
            {video.language?.caption && (
              <span className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded">
                Lang: {video.language.caption.toUpperCase()}
              </span>
            )}
            {video.is_ad && (
              <span className="bg-yellow-900/50 text-yellow-300 text-xs px-2 py-1 rounded">
                Sponsored
              </span>
            )}
            {video.is_ai_generated && (
              <span className="bg-purple-900/50 text-purple-300 text-xs px-2 py-1 rounded">
                AI Generated
              </span>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-3">
            <a
              href={video.url}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary text-sm"
            >
              View on {video.platform}
            </a>
            {video.storage_url && (
              <a href={video.storage_url} download className="btn-primary text-sm">
                Download Video
              </a>
            )}
            {ann && (
              <a
                href={`/api/videos/${video.id}/annotation/download`}
                download
                className="btn-secondary text-sm"
              >
                Download Annotation JSON
              </a>
            )}
          </div>
        </div>
      </div>

      {/* Annotation Content */}
      {isV2 ? (
        <V2AnnotationView annotation={ann} video={video} />
      ) : ann ? (
        <V1FallbackView annotation={ann} video={video} />
      ) : (
        <div className="card">
          <p className="text-gray-500">No annotation data available. Pipeline may still be processing.</p>
        </div>
      )}

      {/* Comments */}
      {video.comments?.length > 0 && (
        <CollapsibleSection title="Comments" count={video.comments.length}>
          <CommentsSection comments={video.comments} />
        </CollapsibleSection>
      )}

      {/* Analytics */}
      {video.analytics && Object.keys(video.analytics).length > 0 && (
        <CollapsibleSection title="Analytics">
          <AnalyticsSection analytics={video.analytics} />
        </CollapsibleSection>
      )}
    </div>
  )
}

/* ── V2 Annotation View (LLM-powered) ─────────────────────────── */

function V2AnnotationView({ annotation, video }) {
  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-200 mb-3">Summary</h2>
        <p className="text-gray-300 leading-relaxed">{annotation.summary}</p>
      </div>

      {/* Scene Timeline */}
      {annotation.scenes?.length > 0 && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-200 mb-4">Scene Timeline</h2>
          <div className="space-y-4">
            {annotation.scenes.map((scene, i) => (
              <div key={i} className="border-l-2 border-blue-500/50 pl-4">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-blue-400 font-mono text-sm">
                    {formatTime(scene.timestamp_sec)}
                  </span>
                </div>
                <p className="text-gray-300 text-sm mb-2">{scene.description}</p>
                {scene.objects_in_context?.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-1">
                    {scene.objects_in_context.map((obj, j) => (
                      <span key={j} className="bg-gray-800 text-gray-400 text-xs px-2 py-0.5 rounded">
                        {obj}
                      </span>
                    ))}
                  </div>
                )}
                {scene.activities?.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {scene.activities.map((act, j) => (
                      <span key={j} className="bg-purple-900/40 text-purple-300 text-xs px-2 py-0.5 rounded">
                        {act}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Activities + Content Tags */}
      {(annotation.primary_activities?.length > 0 || annotation.content_tags?.length > 0) && (
        <div className="card">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {annotation.primary_activities?.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">Activities</h3>
                <div className="flex flex-wrap gap-2">
                  {annotation.primary_activities.map((act) => (
                    <span key={act} className="bg-purple-900/50 text-purple-300 text-sm px-3 py-1 rounded-full">
                      {act}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {annotation.content_tags?.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">Content Tags</h3>
                <div className="flex flex-wrap gap-2">
                  {annotation.content_tags.map((tag) => (
                    <span key={tag} className="bg-blue-900/50 text-blue-300 text-sm px-3 py-1 rounded-full">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Auto Tags (from CLIP + LLM) */}
      {video.tags?.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-400 mb-2">All Tags</h3>
          <div className="flex flex-wrap gap-2">
            {video.tags.map((tag) => (
              <span key={tag} className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Visual Style */}
      {annotation.visual_style && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Visual Style</h3>
          <p className="text-gray-300 text-sm">{annotation.visual_style}</p>
        </div>
      )}

      {/* Transcript Context + Full Transcript */}
      {(annotation.transcript_context || annotation.transcript) && (
        <CollapsibleSection title="Transcript" defaultOpen={false}>
          {annotation.transcript_context && (
            <div className="mb-4">
              <h4 className="text-xs font-medium text-gray-500 mb-1">Context</h4>
              <p className="text-gray-400 text-sm">{annotation.transcript_context}</p>
            </div>
          )}
          {annotation.transcript?.full_text && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 mb-1">
                Full Text
                {annotation.transcript.language && (
                  <span className="ml-2 text-gray-600">({annotation.transcript.language})</span>
                )}
              </h4>
              <p className="text-gray-300 text-sm whitespace-pre-wrap">{annotation.transcript.full_text}</p>
            </div>
          )}
          {annotation.transcript?.segments?.length > 0 && (
            <div className="mt-4">
              <h4 className="text-xs font-medium text-gray-500 mb-2">Timed Segments</h4>
              <div className="space-y-1">
                {annotation.transcript.segments.map((s, i) => (
                  <div key={i} className="flex gap-3 text-sm">
                    <span className="text-blue-400 font-mono w-24 flex-shrink-0">
                      {formatTime(s.start)} - {formatTime(s.end)}
                    </span>
                    <span className="text-gray-300">{s.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CollapsibleSection>
      )}

      {/* Platform Context */}
      {annotation.platform_context && Object.keys(annotation.platform_context).length > 0 && (
        <CollapsibleSection title="Platform Context" defaultOpen={false}>
          <PlatformContextSection context={annotation.platform_context} video={video} />
        </CollapsibleSection>
      )}
    </div>
  )
}

/* ── V1 Fallback View (raw JSON) ──────────────────────────────── */

function V1FallbackView({ annotation, video }) {
  return (
    <div className="card space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-gray-200">Annotation (v1)</h2>
        <span className="bg-yellow-900/50 text-yellow-300 text-xs px-2 py-0.5 rounded">Legacy format</span>
      </div>
      <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap bg-gray-900/50 rounded-lg p-4">
        {JSON.stringify(annotation, null, 2)}
      </pre>
    </div>
  )
}

/* ── Collapsible Section ──────────────────────────────────────── */

function CollapsibleSection({ title, count, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between text-left"
      >
        <h2 className="text-lg font-semibold text-gray-200">
          {title}
          {count != null && <span className="text-gray-500 text-sm font-normal ml-2">({count})</span>}
        </h2>
        <span className="text-gray-500 text-sm">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && <div className="mt-4">{children}</div>}
    </div>
  )
}

/* ── Platform Context Section ─────────────────────────────────── */

function PlatformContextSection({ context, video }) {
  return (
    <div className="space-y-4">
      {context.creator && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 mb-1">Creator</h4>
          <div className="text-sm text-gray-300">
            @{context.creator.username}
            {context.creator.followers != null && (
              <span className="text-gray-500 ml-2">{formatCount(context.creator.followers)} followers</span>
            )}
          </div>
        </div>
      )}
      {context.engagement && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 mb-1">Engagement</h4>
          <div className="flex flex-wrap gap-3 text-sm">
            {Object.entries(context.engagement).map(([k, v]) => (
              <span key={k} className="text-gray-300">
                <span className="text-gray-500 mr-1">{k}</span>
                {formatCount(v)}
              </span>
            ))}
          </div>
        </div>
      )}
      {context.hashtags?.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 mb-1">Hashtags</h4>
          <div className="flex flex-wrap gap-2">
            {context.hashtags.map((tag) => (
              <span key={tag} className="bg-blue-900/30 text-blue-400 text-xs px-2 py-1 rounded">
                #{tag}
              </span>
            ))}
          </div>
        </div>
      )}
      {context.mentions?.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 mb-1">Mentions</h4>
          <div className="flex flex-wrap gap-2">
            {context.mentions.map((m) => (
              <span key={m} className="text-blue-400 text-sm">@{m}</span>
            ))}
          </div>
        </div>
      )}
      {context.music_info && (context.music_info.title || context.music_info.artist) && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 mb-1">Music</h4>
          <span className="text-gray-300 text-sm">
            {context.music_info.title}
            {context.music_info.artist && ` - ${context.music_info.artist}`}
          </span>
        </div>
      )}
      {video.sticker_texts?.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-500 mb-1">Overlay Text (OCR)</h4>
          <div className="flex flex-wrap gap-2">
            {video.sticker_texts.map((t, i) => (
              <span key={i} className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded">{t}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Comments Section ─────────────────────────────────────────── */

function CommentsSection({ comments }) {
  return (
    <div className="space-y-4">
      {comments.map((c, i) => (
        <div key={i} className="border-b border-gray-800 pb-3 last:border-0">
          <div className="flex items-center gap-3 mb-1">
            <span className="font-medium text-gray-200 text-sm">
              {c.username ? `@${c.username}` : 'Anonymous'}
            </span>
            {c.sentiment && <SentimentBadge sentiment={c.sentiment} />}
            {c.posted_at && (
              <span className="text-gray-600 text-xs">
                {new Date(c.posted_at).toLocaleDateString()}
              </span>
            )}
          </div>
          <p className="text-gray-300 text-sm">{c.text}</p>
          <div className="flex gap-4 mt-1 text-xs text-gray-500">
            {c.like_count > 0 && <span>{formatCount(c.like_count)} likes</span>}
            {c.reply_count > 0 && <span>{c.reply_count} replies</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

function SentimentBadge({ sentiment }) {
  const colors = {
    positive: 'bg-green-900/50 text-green-300',
    negative: 'bg-red-900/50 text-red-300',
    neutral: 'bg-gray-700 text-gray-400',
  }
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${colors[sentiment.label] || colors.neutral}`}>
      {sentiment.label} ({sentiment.compound > 0 ? '+' : ''}{sentiment.compound.toFixed(2)})
    </span>
  )
}

/* ── Analytics Section ────────────────────────────────────────── */

function AnalyticsSection({ analytics }) {
  return (
    <div className="space-y-8">
      <SentimentSection data={analytics.comment_sentiment} />
      <EngagementSection data={analytics.engagement} />
      <CategoriesSection data={analytics.content_categories} />
      <BrandSafetySection data={analytics.brand_safety} />
      <AIGenSection data={analytics.ai_generated} />
    </div>
  )
}

function SentimentSection({ data }) {
  if (!data || data.total_analyzed === 0) {
    return (
      <div>
        <h3 className="text-sm font-medium text-gray-300 mb-2">Comment Sentiment</h3>
        <p className="text-gray-500 text-sm">No comments analyzed</p>
      </div>
    )
  }

  const { distribution, avg_compound, total_analyzed, skipped_non_english, top_positive, top_negative } = data
  const total = distribution.positive + distribution.neutral + distribution.negative
  const pctPos = total > 0 ? (distribution.positive / total * 100).toFixed(0) : 0
  const pctNeu = total > 0 ? (distribution.neutral / total * 100).toFixed(0) : 0
  const pctNeg = total > 0 ? (distribution.negative / total * 100).toFixed(0) : 0

  return (
    <div>
      <h3 className="text-sm font-medium text-gray-300 mb-3">Comment Sentiment</h3>
      <div className="flex items-center gap-2 mb-3 text-xs text-gray-400">
        <span>{total_analyzed} analyzed</span>
        {skipped_non_english > 0 && <span>({skipped_non_english} non-English skipped)</span>}
        <span className="ml-auto">Avg compound: <span className={avg_compound >= 0 ? 'text-green-400' : 'text-red-400'}>{avg_compound > 0 ? '+' : ''}{avg_compound.toFixed(3)}</span></span>
      </div>
      <div className="flex h-6 rounded overflow-hidden mb-3">
        {distribution.positive > 0 && (
          <div className="bg-green-600 flex items-center justify-center text-xs text-white" style={{ width: `${pctPos}%` }}>{pctPos}%</div>
        )}
        {distribution.neutral > 0 && (
          <div className="bg-gray-600 flex items-center justify-center text-xs text-white" style={{ width: `${pctNeu}%` }}>{pctNeu}%</div>
        )}
        {distribution.negative > 0 && (
          <div className="bg-red-600 flex items-center justify-center text-xs text-white" style={{ width: `${pctNeg}%` }}>{pctNeg}%</div>
        )}
      </div>
      <div className="flex gap-4 text-xs text-gray-400 mb-4">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-600" /> Positive ({distribution.positive})</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-600" /> Neutral ({distribution.neutral})</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-600" /> Negative ({distribution.negative})</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {top_positive && (
          <div className="border border-green-900/50 rounded p-3">
            <span className="text-xs text-green-400 block mb-1">Most Positive ({top_positive.compound > 0 ? '+' : ''}{top_positive.compound.toFixed(2)})</span>
            <p className="text-gray-300 text-sm italic">"{top_positive.text}"</p>
          </div>
        )}
        {top_negative && (
          <div className="border border-red-900/50 rounded p-3">
            <span className="text-xs text-red-400 block mb-1">Most Negative ({top_negative.compound.toFixed(2)})</span>
            <p className="text-gray-300 text-sm italic">"{top_negative.text}"</p>
          </div>
        )}
      </div>
    </div>
  )
}

function EngagementSection({ data }) {
  if (!data || !data.rates) {
    return (
      <div>
        <h3 className="text-sm font-medium text-gray-300 mb-2">Engagement Benchmarks</h3>
        <p className="text-gray-500 text-sm">No engagement data available</p>
      </div>
    )
  }

  const { rates, benchmarks, performance_tier } = data
  const tierColors = {
    viral: 'bg-green-500 text-white',
    excellent: 'bg-blue-500 text-white',
    good: 'bg-teal-500 text-white',
    average: 'bg-gray-500 text-white',
    below_average: 'bg-red-500 text-white',
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <h3 className="text-sm font-medium text-gray-300">Engagement Benchmarks</h3>
        <span className={`text-xs px-2 py-0.5 rounded font-medium ${tierColors[performance_tier] || tierColors.average}`}>
          {performance_tier?.replace('_', ' ')}
        </span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
        {[
          ['Engagement', rates.engagement_rate],
          ['Like Rate', rates.like_rate],
          ['Comment Rate', rates.comment_rate],
          ['Share Rate', rates.share_rate],
          ['Save Rate', rates.save_rate],
        ].map(([label, rate]) => (
          <div key={label} className="bg-gray-800/50 rounded p-2 text-center">
            <span className="text-xs text-gray-500 block">{label}</span>
            <span className="text-sm font-mono text-gray-200">{(rate * 100).toFixed(2)}%</span>
          </div>
        ))}
      </div>
      {benchmarks && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {benchmarks.vs_creator && benchmarks.vs_creator.verdict !== 'insufficient_data' && (
            <div className="bg-gray-800/50 rounded p-3">
              <span className="text-xs text-gray-500 block mb-1">vs Creator Average</span>
              <span className="text-sm text-gray-200 font-mono">{benchmarks.vs_creator.ratio}x</span>
              <span className="text-xs text-gray-500 ml-2">({benchmarks.vs_creator.sample_size} videos)</span>
              <VerdictBadge verdict={benchmarks.vs_creator.verdict} />
            </div>
          )}
          {benchmarks.vs_platform && benchmarks.vs_platform.verdict !== 'insufficient_data' && (
            <div className="bg-gray-800/50 rounded p-3">
              <span className="text-xs text-gray-500 block mb-1">vs Platform Average</span>
              <span className="text-sm text-gray-200 font-mono">{benchmarks.vs_platform.ratio}x</span>
              <span className="text-xs text-gray-500 ml-2">({benchmarks.vs_platform.sample_size} videos)</span>
              <VerdictBadge verdict={benchmarks.vs_platform.verdict} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function VerdictBadge({ verdict }) {
  const colors = {
    above_average: 'text-green-400',
    average: 'text-gray-400',
    below_average: 'text-red-400',
  }
  return (
    <span className={`text-xs ml-2 ${colors[verdict] || colors.average}`}>
      {verdict?.replace('_', ' ')}
    </span>
  )
}

function CategoriesSection({ data }) {
  if (!data?.length) {
    return (
      <div>
        <h3 className="text-sm font-medium text-gray-300 mb-2">Content Categories</h3>
        <p className="text-gray-500 text-sm">No categories detected</p>
      </div>
    )
  }

  return (
    <div>
      <h3 className="text-sm font-medium text-gray-300 mb-3">Content Categories</h3>
      <div className="space-y-2">
        {data.map((cat) => (
          <div key={cat.category} className="flex items-center gap-3">
            <span className="text-sm text-gray-300 w-24">{cat.category}</span>
            <div className="flex-1 bg-gray-800 rounded-full h-4 overflow-hidden">
              <div className="bg-blue-500 h-full rounded-full" style={{ width: `${(cat.confidence * 100).toFixed(0)}%` }} />
            </div>
            <span className="text-xs text-gray-400 font-mono w-12 text-right">{(cat.confidence * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function BrandSafetySection({ data }) {
  if (!data) {
    return (
      <div>
        <h3 className="text-sm font-medium text-gray-300 mb-2">Brand Safety</h3>
        <p className="text-gray-500 text-sm">No brand safety data</p>
      </div>
    )
  }

  const tierColors = {
    safe: 'bg-green-900/50 text-green-300',
    low_risk: 'bg-yellow-900/50 text-yellow-300',
    medium_risk: 'bg-orange-900/50 text-orange-300',
    high_risk: 'bg-red-900/50 text-red-300',
  }
  const severityColors = {
    high: 'text-red-400',
    medium: 'text-orange-400',
    low: 'text-yellow-400',
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <h3 className="text-sm font-medium text-gray-300">Brand Safety</h3>
        <span className={`text-xs px-2 py-0.5 rounded font-medium ${tierColors[data.tier] || tierColors.safe}`}>
          {data.tier?.replace('_', ' ')}
        </span>
        <span className="text-sm text-gray-400 font-mono">{data.score.toFixed(2)}</span>
      </div>
      {data.flags?.length > 0 ? (
        <div className="space-y-1">
          {data.flags.map((flag, i) => (
            <div key={i} className="flex items-center gap-3 text-sm">
              <span className={`text-xs font-medium ${severityColors[flag.severity] || 'text-gray-400'}`}>{flag.severity}</span>
              <span className="text-gray-400">{flag.source}</span>
              <span className="text-gray-300">"{flag.keyword}"</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500 text-sm">No safety flags detected</p>
      )}
    </div>
  )
}

function AIGenSection({ data }) {
  if (!data) {
    return (
      <div>
        <h3 className="text-sm font-medium text-gray-300 mb-2">AI-Generated Detection</h3>
        <p className="text-gray-500 text-sm">No AI detection data</p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <h3 className="text-sm font-medium text-gray-300">AI-Generated Detection</h3>
        <span className={`text-xs px-2 py-0.5 rounded font-medium ${
          data.is_ai_generated ? 'bg-purple-900/50 text-purple-300' : 'bg-green-900/50 text-green-300'
        }`}>
          {data.is_ai_generated ? 'AI Generated' : 'Human Created'}
        </span>
        <span className="text-sm text-gray-400 font-mono">confidence: {(data.confidence * 100).toFixed(0)}%</span>
      </div>
      {data.signals && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="bg-gray-800/50 rounded p-2 text-center">
            <span className="text-xs text-gray-500 block">Keyword Match</span>
            <span className={`text-sm ${data.signals.keyword_match ? 'text-red-400' : 'text-green-400'}`}>
              {data.signals.keyword_match ? 'Yes' : 'No'}
            </span>
          </div>
          {data.signals.mean_cosine_similarity != null && (
            <div className="bg-gray-800/50 rounded p-2 text-center">
              <span className="text-xs text-gray-500 block">Mean Frame Similarity</span>
              <span className="text-sm text-gray-200 font-mono">{data.signals.mean_cosine_similarity.toFixed(4)}</span>
            </div>
          )}
          {data.signals.embedding_variance != null && (
            <div className="bg-gray-800/50 rounded p-2 text-center">
              <span className="text-xs text-gray-500 block">Embedding Variance</span>
              <span className="text-sm text-gray-200 font-mono">{data.signals.embedding_variance.toFixed(6)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Shared Components ────────────────────────────────────────── */

function GDPRBanner({ gdprFlags, gdprStatus }) {
  if (!gdprFlags) return null
  const status = gdprFlags.status

  if (status === 'clean') {
    return (
      <div className="bg-green-900/20 border border-green-800 rounded-lg p-4 flex items-start gap-3">
        <span className="text-green-400 text-lg">&#10003;</span>
        <div>
          <span className="text-green-300 font-medium text-sm">GDPR Compliant</span>
          <p className="text-green-400/70 text-xs mt-0.5">{gdprStatus}</p>
        </div>
      </div>
    )
  }

  if (status === 'blocked') {
    return (
      <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 flex items-start gap-3">
        <span className="text-red-400 text-lg">&#9888;</span>
        <div>
          <span className="text-red-300 font-medium text-sm">Not GDPR Compliant</span>
          <p className="text-red-400/70 text-xs mt-0.5">{gdprStatus}</p>
          {gdprFlags.pii_types?.length > 0 && (
            <div className="flex gap-1 mt-2">
              {gdprFlags.pii_types.map((t) => (
                <span key={t} className="bg-red-900/50 text-red-300 text-xs px-1.5 py-0.5 rounded">{t}</span>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  if (status === 'unverified') {
    return (
      <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg p-4 flex items-start gap-3">
        <span className="text-yellow-400 text-lg">&#9888;</span>
        <div>
          <span className="text-yellow-300 font-medium text-sm">GDPR Unverified</span>
          <p className="text-yellow-400/70 text-xs mt-0.5">{gdprStatus}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-orange-900/20 border border-orange-800 rounded-lg p-4 flex items-start gap-3">
      <span className="text-orange-400 text-lg">&#9888;</span>
      <div>
        <span className="text-orange-300 font-medium text-sm">GDPR Check Error</span>
        <p className="text-orange-400/70 text-xs mt-0.5">{gdprStatus}</p>
      </div>
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

function formatTime(seconds) {
  if (seconds == null) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function formatCount(n) {
  if (n == null) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return String(n)
}

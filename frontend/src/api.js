const BASE = ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    // Try to extract structured error message
    try {
      const json = JSON.parse(text)
      const detail = json.detail
      if (detail && typeof detail === 'object' && detail.message) {
        const err = new Error(detail.message)
        err.code = detail.error || 'unknown'
        err.status = res.status
        err.detail = detail
        throw err
      }
    } catch (e) {
      if (e.code) throw e  // re-throw our structured error
    }
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

export async function getStats() {
  return request('/stats')
}

export async function getHealth() {
  return request('/health')
}

export async function createIngestion(query) {
  return request('/api/ingest', {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
}

export async function searchVideos(q, { limit = 20, platform, minSimilarity } = {}) {
  const params = new URLSearchParams({ q, limit })
  if (platform) params.set('platform', platform)
  if (minSimilarity) params.set('min_similarity', minSimilarity)
  return request(`/api/search?${params}`)
}

export async function listVideos({ page = 1, pageSize = 20, platform, status, tag } = {}) {
  const params = new URLSearchParams({ page, page_size: pageSize })
  if (platform) params.set('platform', platform)
  if (status) params.set('status', status)
  if (tag) params.set('tag', tag)
  return request(`/api/videos?${params}`)
}

export async function getVideo(id) {
  return request(`/api/videos/${id}`)
}

export async function listTasks({ status, limit = 50 } = {}) {
  const params = new URLSearchParams({ limit })
  if (status) params.set('status', status)
  return request(`/api/tasks?${params}`)
}

export async function getTask(id) {
  return request(`/api/tasks/${id}`)
}

export async function getTaskVideos(id) {
  return request(`/api/tasks/${id}/videos`)
}

export async function createDataset(name, description = '', filters = {}) {
  return request('/api/datasets', {
    method: 'POST',
    body: JSON.stringify({ name, description, filters }),
  })
}

export async function listDatasets() {
  return request('/api/datasets')
}

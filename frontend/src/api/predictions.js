const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function parseResponse(response) {
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || 'Сервис временно недоступен. Попробуйте ещё раз.')
  return body
}

async function request(url, options) {
  try {
    return await parseResponse(await fetch(url, options))
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error('Не удалось подключиться к сервису оценки. Проверьте, что API запущен.')
    }
    throw error
  }
}

export async function createPrediction({ transportId, file }) {
  const data = new FormData()
  data.append('transport_id', transportId.trim())
  data.append('image', file)
  return request(`${API_BASE}/api/v1/predictions`, { method: 'POST', body: data })
}

export async function getPrediction(transportId) {
  return request(`${API_BASE}/api/v1/predictions/${encodeURIComponent(transportId.trim())}`)
}

import { afterEach, describe, expect, it, vi } from 'vitest'

import { getPrediction } from './predictions'

afterEach(() => vi.restoreAllMocks())

describe('prediction API', () => {
  it('replaces a network failure with a user-facing message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))

    await expect(getPrediction('TR-1')).rejects.toThrow('Не удалось подключиться к сервису оценки.')
  })
})

import { describe, expect, it } from 'vitest'

import { validatePrediction } from './validation'

describe('validatePrediction', () => {
  it('requires a transport number', () => {
    expect(validatePrediction({ transportId: ' ', file: new File(['x'], 'truck.jpg', { type: 'image/jpeg' }) }))
      .toBe('Введите номер перевозки.')
  })

  it('rejects unsupported photo formats', () => {
    expect(validatePrediction({ transportId: 'TR-2026-001245', file: new File(['x'], 'truck.gif', { type: 'image/gif' }) }))
      .toBe('Поддерживаются изображения JPG, PNG и WEBP.')
  })
})

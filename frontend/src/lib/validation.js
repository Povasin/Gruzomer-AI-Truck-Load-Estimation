const SUPPORTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp'])

export function validatePrediction({ transportId, file }) {
  if (!transportId.trim()) return 'Введите номер перевозки.'
  if (!file) return 'Добавьте фотографию грузового отсека.'
  if (!SUPPORTED_TYPES.has(file.type)) return 'Поддерживаются изображения JPG, PNG и WEBP.'
  return null
}

import { useEffect, useState } from 'react'

import { createPrediction, getPrediction } from './api/predictions'
import { validatePrediction } from './lib/validation'

const tabs = [
  { id: 'new', label: 'Новая оценка' },
  { id: 'search', label: 'Найти перевозку' },
]

const cargoTypeLabels = {
  blue_open_containers: 'Открытые контейнеры',
  pallets: 'Паллеты',
  blue_full_container: 'Закрытый контейнер',
  boxes: 'Коробки',
  mixed: 'Смешанный груз',
}

function ResultCard({ result, onReset }) {
  return (
    <section className="result-card" aria-live="polite">
      <p className="section-kicker">Результат анализа</p>
      <div className="result-number">{Math.round(result.load_pct)}<span>%</span></div>
      <div className="progress-track" aria-label={`Загрузка ${Math.round(result.load_pct)} процентов`}>
        <div className="progress-fill" style={{ width: `${result.load_pct}%` }} />
      </div>
      <div className="result-row"><span>Перевозка</span><strong>{result.transport_id}</strong></div>
      <div className="result-row"><span>Тип груза</span><strong>{cargoTypeLabels[result.cargo_type] ?? result.cargo_type ?? 'Не определён'}</strong></div>
      {result.backbone && <div className="result-row"><span>Модель</span><strong>{result.backbone}</strong></div>}
      <p className="success-note">Результат сохранён</p>
      <button className="secondary-button" onClick={onReset}>Оценить другое фото</button>
    </section>
  )
}

export default function App() {
  const [tab, setTab] = useState('new')
  const [transportId, setTransportId] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (!file) { setPreview(null); return undefined }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const reset = () => { setFile(null); setResult(null); setError(''); setTransportId('') }

  async function submitPrediction(event) {
    event.preventDefault()
    const message = validatePrediction({ transportId, file })
    if (message) return setError(message)
    setLoading(true); setError(''); setResult(null)
    try { setResult(await createPrediction({ transportId, file })) }
    catch (cause) { setError(cause.message) }
    finally { setLoading(false) }
  }

  async function submitSearch(event) {
    event.preventDefault()
    if (!transportId.trim()) return setError('Введите номер перевозки.')
    setLoading(true); setError(''); setResult(null)
    try { setResult(await getPrediction(transportId)) }
    catch (cause) { setError(cause.message) }
    finally { setLoading(false) }
  }

  return (
    <main className="page-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="Грузомер — на главную"><span className="brand-mark">G</span> ГРУЗОМЕР</a>
        <nav className="site-nav" aria-label="Основная навигация">
          <a href="#workspace">Оценить загрузку</a>
          <a href="#workspace">Найти перевозку</a>
        </nav>
        <span className="header-status">MVP · Фото → оценка</span>
      </header>
      <div className="hero-grid">
        <section className="intro">
          <p className="section-kicker">Контроль загрузки</p>
          <h1>Увидеть заполнение кузова — в одном кадре.</h1>
          <p>Загрузите фотографию грузового отсека и сохраните оценку за несколько секунд.</p>
          <div className="hero-details" aria-label="Поддерживаемые возможности">
            <span>Фото до 10 МБ</span><span>JPG · PNG · WEBP</span>
          </div>
        </section>
        <aside className="hero-illustration" aria-label="Иллюстрация оценки загрузки грузового транспорта">
          <div className="stage-status"><i />Анализ в реальном времени</div>
          <div className="stage-note"><b>Фото</b><span>→</span><b>Оценка</b></div>
          <img src="/illustrations/cargo-load-hero.png" alt="Грузовик с заполненным коробками кузовом и сотрудник логистики" />
        </aside>
      </div>
      <section className="workspace" id="workspace">
        <div className="tabs" role="tablist" aria-label="Действия с перевозкой">
          {tabs.map((item) => <button key={item.id} role="tab" aria-selected={tab === item.id} className={tab === item.id ? 'active' : ''} onClick={() => { setTab(item.id); setError(''); setResult(null) }}>{item.label}</button>)}
        </div>
        {result ? <ResultCard result={result} onReset={reset} /> : tab === 'new' ? (
          <form className="form" onSubmit={submitPrediction}>
            <label>Номер перевозки<input value={transportId} maxLength="80" onChange={(event) => setTransportId(event.target.value)} placeholder="TR-2026-001245" /></label>
            <div className="field-label">Фотография кузова</div>
            <label className={`dropzone ${preview ? 'has-preview' : ''}`}>
              <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setError('') }} />
              {preview ? <img src={preview} alt="Предпросмотр кузова" /> : <><strong>Перетащите фото сюда</strong><span>или выберите файл JPG, PNG или WEBP</span></>}
            </label>
            {file && <p className="file-name">Выбран файл: {file.name}</p>}
            {error && <p className="error" role="alert">{error}</p>}
            <button className="primary-button" disabled={loading}>{loading ? 'Анализируем грузовой отсек…' : 'Оценить загрузку'}</button>
          </form>
        ) : (
          <form className="form search-form" onSubmit={submitSearch}>
            <p className="form-copy">Введите номер, чтобы найти сохранённую оценку загрузки.</p>
            <label>Номер перевозки<input value={transportId} maxLength="80" onChange={(event) => setTransportId(event.target.value)} placeholder="TR-2026-001245" /></label>
            {error && <p className="error" role="alert">{error}</p>}
            <button className="primary-button" disabled={loading}>{loading ? 'Ищем результат…' : 'Найти результат'}</button>
          </form>
        )}
      </section>
    </main>
  )
}

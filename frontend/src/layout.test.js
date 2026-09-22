import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync(new URL('./index.css', import.meta.url), 'utf8')

describe('hero workspace layout', () => {
  it('lets the workspace span the full hero width on desktop', () => {
    expect(css).toMatch(/\.workspace\s*\{[^}]*max-width:\s*none/)
  })

  it('keeps the larger hero illustration within its grid column on desktop', () => {
    expect(css).toMatch(/\.hero-illustration\s*\{[^}]*overflow:\s*hidden/)
    expect(css).toMatch(/\.hero-illustration img\s*\{[^}]*width:\s*clamp\(/)
    expect(css).toMatch(/\.hero-illustration img\s*\{[^}]*transform:\s*translate\(0/)
  })
})

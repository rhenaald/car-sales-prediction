// ── CSV Parser ────────────────────────────────────────────────────────────────
export function parseCSV(text) {
  const lines = text.trim().split('\n')
  const headers = lines[0].split(',').map(h => h.trim())
  return lines.slice(1).map(line => {
    const vals = line.split(',')
    const row = {}
    headers.forEach((h, i) => { row[h] = vals[i]?.trim() ?? '' })
    return row
  })
}

// ── Feature columns used for regression ───────────────────────────────────────
export const FEATURE_KEYS = [
  'Engine_size', 'Horsepower', 'Wheelbase',
  'Width', 'Length', 'Curb_weight',
  'Fuel_capacity', 'Fuel_efficiency'
]
export const TARGET_KEY = 'Price_in_thousands'

// ── Filter rows that have all required numeric values ─────────────────────────
export function getValidRows(rows) {
  return rows.filter(r =>
    [...FEATURE_KEYS, TARGET_KEY].every(k => r[k] && r[k] !== '' && !isNaN(parseFloat(r[k])))
  )
}

// ── Matrix multiply: A (m×n) × B (n×p) ───────────────────────────────────────
function matMul(A, B) {
  const m = A.length, n = B.length, p = B[0].length
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: p }, (_, j) =>
      A[i].reduce((s, _, k) => s + A[i][k] * B[k][j], 0)
    )
  )
}

// ── Transpose ─────────────────────────────────────────────────────────────────
function transpose(M) {
  return M[0].map((_, j) => M.map(row => row[j]))
}

// ── Gauss-Jordan matrix inverse ───────────────────────────────────────────────
function inverse(M) {
  const n = M.length
  const aug = M.map((row, i) => {
    const id = Array(n).fill(0); id[i] = 1
    return [...row, ...id]
  })
  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]]
    const div = aug[col][col]
    aug[col] = aug[col].map(v => v / div)
    for (let row = 0; row < n; row++) {
      if (row === col) continue
      const factor = aug[row][col]
      aug[row] = aug[row].map((v, j) => v - factor * aug[col][j])
    }
  }
  return aug.map(row => row.slice(n))
}

// ── Train Linear Regression using Normal Equation: θ = (XᵀX)⁻¹ Xᵀy ──────────
export function trainLinearRegression(validRows) {
  const n = validRows.length

  // Build design matrix X (with bias column) and target vector y
  const X = validRows.map(r => [1, ...FEATURE_KEYS.map(k => parseFloat(r[k]))])
  const y = validRows.map(r => [parseFloat(r[TARGET_KEY])])

  const Xt = transpose(X)
  const XtX = matMul(Xt, X)
  const XtX_inv = inverse(XtX)
  const Xty = matMul(Xt, y)
  const theta = matMul(XtX_inv, Xty).map(r => r[0])   // [bias, w1, w2, ...]

  // ── Compute R² ──────────────────────────────────────────────────────────────
  const yFlat = y.map(r => r[0])
  const yMean = yFlat.reduce((a, b) => a + b, 0) / n
  const yPred = X.map(row => row.reduce((s, v, i) => s + v * theta[i], 0))
  const ssRes = yFlat.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0)
  const ssTot = yFlat.reduce((s, v) => s + (v - yMean) ** 2, 0)
  const r2 = 1 - ssRes / ssTot

  // ── Per-feature stats (for summary display) ──────────────────────────────────
  const featureStats = {}
  FEATURE_KEYS.forEach(k => {
    const vals = validRows.map(r => parseFloat(r[k]))
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    featureStats[k] = { min, max, mean }
  })

  return { theta, r2, n, featureStats }
}

// ── Predict price given inputs and trained theta ───────────────────────────────
export function predictPrice(inputs, theta) {
  const features = [1, ...FEATURE_KEYS.map(k => parseFloat(inputs[k] || 0))]
  const price = features.reduce((s, v, i) => s + v * theta[i], 0)
  return Math.max(price, 0)
}

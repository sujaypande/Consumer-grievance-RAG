// Hardcoded because this is a local, single-machine project (backend
// always runs on localhost:8000) -- not worth an env var for something
// that never changes.
const API_BASE = "http://localhost:8000/api/v1"

export async function askQuestion(query) {
  const response = await fetch(`${API_BASE}/langchain/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  })
  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`)
  }
  return response.json()
}

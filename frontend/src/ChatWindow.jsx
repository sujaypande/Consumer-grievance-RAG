import { useState } from "react"
import { askQuestion } from "./api"

function ChatWindow() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(event) {
    event.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    setMessages((prev) => [...prev, { role: "user", text: query }])
    setInput("")
    setLoading(true)
    setError(null)

    try {
      const result = await askQuestion(query)
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: result.answer, sources: result.sources },
      ])
    } catch (err) {
      setError("Something went wrong asking the backend. Is it running on localhost:8000?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-window">
      <h2>Consumer Grievance Assistant</h2>

      <div className="messages">
        {messages.length === 0 && (
          <p className="empty-hint">Ask a question about the Consumer Protection Act, 2019.</p>
        )}
        {messages.map((message, i) => (
          <div key={i} className={`message ${message.role}`}>
            <p>{message.text}</p>
            {message.sources && message.sources.length > 0 && (
              <p className="sources">Sources: {message.sources.join(", ")}</p>
            )}
          </div>
        ))}
        {loading && <p className="empty-hint">Thinking...</p>}
      </div>

      {error && <p className="error">{error}</p>}

      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. What is the time limit for filing a complaint?"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Ask
        </button>
      </form>
    </div>
  )
}

export default ChatWindow

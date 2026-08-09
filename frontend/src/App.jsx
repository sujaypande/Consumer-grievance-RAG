import ChatWindow from "./ChatWindow"
import "./App.css"

function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Consumer Grievance RAG</h1>
        <p className="tagline">Hybrid search + reranked RAG over the Consumer Protection Act, 2019</p>
      </aside>

      <main>
        <ChatWindow />
      </main>
    </div>
  )
}

export default App

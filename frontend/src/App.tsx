import { useEffect, useState } from 'react'
import { JarvisWave } from './components/JarvisWave'
import { STATE_LABELS, type PipelineState, type StatusEvent } from './lib/types'
import { StatusSocket } from './lib/ws'
import './App.css'

const INITIAL: StatusEvent = {
  state: 'idle',
  amplitude: 0.08,
  message: '',
}

export default function App() {
  const [status, setStatus] = useState<StatusEvent>(INITIAL)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const socket = new StatusSocket(setStatus, setConnected)
    socket.connect()
    return () => socket.disconnect()
  }, [])

  const state: PipelineState = status.state

  return (
    <div className={`app state-${state}`}>
      <div className="bg-glow" aria-hidden />
      <header className="brand">
        <h1>JARVIS</h1>
        <p className="tagline">ASSISTANT INTERFACE</p>
      </header>

      <main className="stage">
        <JarvisWave state={state} amplitude={status.amplitude} />
      </main>

      <footer className="hud">
        <div className={`status-label ${connected ? 'online' : 'offline'}`}>
          <span className="dot" />
          {connected ? STATE_LABELS[state] : 'SIN CONEXIÓN'}
        </div>
        {status.message ? <p className="status-msg">{status.message}</p> : null}
      </footer>
    </div>
  )
}

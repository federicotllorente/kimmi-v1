export type PipelineState = 'idle' | 'listening' | 'processing' | 'responding'

export type StatusEvent = {
  state: PipelineState
  amplitude: number
  message: string
}

export const STATE_LABELS: Record<PipelineState, string> = {
  idle: 'EN ESPERA',
  listening: 'ESCUCHANDO',
  processing: 'PROCESANDO',
  responding: 'RESPONDIENDO',
}

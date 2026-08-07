import { useMemo, useRef, type ReactElement } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { PipelineState } from '../lib/types'

type KimmiWaveProps = {
  state: PipelineState
  amplitude: number
}

const BAR_COUNT = 64
const CYAN = '#00d4ff'
const DEEP = '#041018'

function stateMotion(state: PipelineState, t: number, amp: number, i: number): number {
  const phase = (i / BAR_COUNT) * Math.PI * 2
  if (state === 'idle') {
    return 0.15 + amp * 0.5 + 0.08 * Math.sin(t * 1.2 + phase)
  }
  if (state === 'listening') {
    return 0.25 + amp * (0.7 + 0.35 * Math.sin(t * 8 + phase * 3))
  }
  if (state === 'processing') {
    const sweep = Math.sin(t * 4.5 - phase * 2)
    return 0.2 + amp * (0.55 + 0.35 * sweep)
  }
  // responding
  return 0.3 + amp * (0.85 + 0.4 * Math.sin(t * 6 + phase * 1.5))
}

function RadialBars({ state, amplitude }: KimmiWaveProps) {
  const group = useRef<THREE.Group>(null)
  const meshes = useRef<THREE.Mesh[]>([])
  const ampRef = useRef(amplitude)
  const stateRef = useRef(state)
  ampRef.current = amplitude
  stateRef.current = state

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: CYAN,
        emissive: CYAN,
        emissiveIntensity: 1.4,
        metalness: 0.35,
        roughness: 0.25,
        transparent: true,
        opacity: 0.92,
      }),
    [],
  )

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const amp = ampRef.current
    const st = stateRef.current

    if (group.current) {
      const spin =
        st === 'idle' ? 0.08 : st === 'listening' ? 0.35 : st === 'processing' ? 0.55 : 0.9
      group.current.rotation.z = t * spin * 0.15
      group.current.rotation.x = Math.sin(t * 0.4) * 0.08
    }

    meshes.current.forEach((mesh, i) => {
      if (!mesh) return
      const h = Math.max(0.05, stateMotion(st, t, amp, i))
      mesh.scale.y = h * 2.2
      mesh.position.y = h * 1.1
      const mat = mesh.material as THREE.MeshStandardMaterial
      mat.emissiveIntensity = 0.8 + amp * (st === 'responding' ? 2.2 : 1.4)
      mat.opacity = 0.55 + amp * 0.4
    })
  })

  const bars = useMemo(() => {
    const items: ReactElement[] = []
    for (let i = 0; i < BAR_COUNT; i++) {
      const angle = (i / BAR_COUNT) * Math.PI * 2
      const radius = 1.55
      items.push(
        <mesh
          key={i}
          ref={(el) => {
            if (el) meshes.current[i] = el
          }}
          position={[Math.cos(angle) * radius, 0.2, Math.sin(angle) * radius]}
          rotation={[0, -angle, 0]}
          material={material}
        >
          <boxGeometry args={[0.06, 1, 0.06]} />
        </mesh>,
      )
    }
    return items
  }, [material])

  return <group ref={group}>{bars}</group>
}

function CoreOrb({ amplitude, state }: KimmiWaveProps) {
  const mesh = useRef<THREE.Mesh>(null)
  const glow = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const pulse =
      state === 'idle'
        ? 0.85 + 0.08 * Math.sin(t * 1.5)
        : 0.9 + amplitude * 0.35 + 0.05 * Math.sin(t * 6)

    if (mesh.current) {
      mesh.current.scale.setScalar(pulse)
    }
    if (glow.current) {
      glow.current.scale.setScalar(pulse * (1.4 + amplitude * 0.8))
      const mat = glow.current.material as THREE.MeshBasicMaterial
      mat.opacity = 0.12 + amplitude * 0.28
    }
  })

  return (
    <group>
      <mesh ref={mesh}>
        <sphereGeometry args={[0.28, 32, 32]} />
        <meshStandardMaterial
          color={CYAN}
          emissive={CYAN}
          emissiveIntensity={2}
          metalness={0.6}
          roughness={0.2}
        />
      </mesh>
      <mesh ref={glow}>
        <sphereGeometry args={[0.28, 24, 24]} />
        <meshBasicMaterial color={CYAN} transparent opacity={0.2} depthWrite={false} />
      </mesh>
    </group>
  )
}

function Rings({ state, amplitude }: KimmiWaveProps) {
  const outer = useRef<THREE.Mesh>(null)
  const mid = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    if (outer.current) {
      outer.current.rotation.z = t * (state === 'responding' ? 0.6 : 0.2)
      outer.current.rotation.x = Math.PI / 2 + Math.sin(t * 0.3) * 0.05
    }
    if (mid.current) {
      mid.current.rotation.z = -t * 0.35
      mid.current.rotation.x = Math.PI / 2
      const mat = mid.current.material as THREE.MeshBasicMaterial
      mat.opacity = 0.25 + amplitude * 0.35
    }
  })

  return (
    <>
      <mesh ref={outer}>
        <torusGeometry args={[2.15, 0.012, 12, 128]} />
        <meshBasicMaterial color={CYAN} transparent opacity={0.45} />
      </mesh>
      <mesh ref={mid}>
        <torusGeometry args={[1.85, 0.008, 12, 96]} />
        <meshBasicMaterial color="#7ae7ff" transparent opacity={0.3} />
      </mesh>
    </>
  )
}

function Scene({ state, amplitude }: KimmiWaveProps) {
  return (
    <>
      <color attach="background" args={[DEEP]} />
      <ambientLight intensity={0.35} />
      <pointLight position={[0, 2, 4]} intensity={1.4} color="#6fdfff" />
      <pointLight position={[0, -2, -3]} intensity={0.6} color="#1a6cff" />
      <RadialBars state={state} amplitude={amplitude} />
      <CoreOrb state={state} amplitude={amplitude} />
      <Rings state={state} amplitude={amplitude} />
      <OrbitControls enableZoom={false} enablePan={false} autoRotate={false} />
    </>
  )
}

export function KimmiWave({ state, amplitude }: KimmiWaveProps) {
  return (
    <div className="wave-canvas">
      <Canvas camera={{ position: [0, 1.4, 4.2], fov: 42 }} dpr={[1, 1.75]}>
        <Scene state={state} amplitude={amplitude} />
      </Canvas>
    </div>
  )
}

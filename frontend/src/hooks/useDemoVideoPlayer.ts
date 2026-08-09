import { useCallback, useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'

// Fallback only — the real URLs should come from VITE_DEMO_VIDEO_URL /
// VITE_DEMO_START_URL.
const DEFAULT_DEMO_VIDEO_URL = 'http://localhost:8000/api/demo/video'
const DEFAULT_DEMO_START_URL = 'http://localhost:8000/api/demo/start'

export const DEMO_VIDEO_URL = import.meta.env.VITE_DEMO_VIDEO_URL ?? DEFAULT_DEMO_VIDEO_URL
const DEMO_START_URL = import.meta.env.VITE_DEMO_START_URL ?? DEFAULT_DEMO_START_URL

export interface UseDemoVideoPlayerResult {
  videoRef: RefObject<HTMLVideoElement | null>
  /** Whether Start Demo has ever been pressed — gates the autoplay-blocked overlay. */
  hasStarted: boolean
  isPlaying: boolean
  isMuted: boolean
  volume: number
  /** First play: requires a genuine user gesture (browsers block autoplay with sound). */
  start: () => void
  togglePlay: () => void
  /** Same reset-to-start as start(), for use once playback is already underway. */
  restart: () => void
  toggleMute: () => void
  setVolume: (value: number) => void
}

/**
 * Owns the demo <video> element's playback state and the "Start Demo"/
 * "Restart" reset-to-t=0 flow. Resetting is two things happening together:
 * the browser element seeking to 0 and playing, and a POST to the backend
 * that seeks services/camera.py's and services/tensorflow_audio.py's own
 * playback position back to the start of the same file — see
 * routes/demo.py. The POST is fired without being awaited so it runs
 * concurrently with (rather than delaying) the local seek+play; see the
 * integration report for the resulting synchronization limitations.
 */
export function useDemoVideoPlayer(): UseDemoVideoPlayerResult {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [hasStarted, setHasStarted] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [volume, setVolumeState] = useState(1)

  const resetBackendSession = useCallback(() => {
    fetch(DEMO_START_URL, { method: 'POST' }).catch(() => {
      // Best-effort: the video still plays locally even if the backend
      // reset request fails (e.g. backend briefly unreachable).
    })
  }, [])

  const playFromStart = useCallback(() => {
    resetBackendSession()
    const video = videoRef.current
    if (!video) return
    video.currentTime = 0
    video.play().catch(() => {
      // Autoplay-with-sound can still be rejected in rare cases even from a
      // click handler — the Start Demo overlay stays available to retry.
    })
  }, [resetBackendSession])

  const start = useCallback(() => {
    setHasStarted(true)
    playFromStart()
  }, [playFromStart])

  const restart = useCallback(() => {
    playFromStart()
  }, [playFromStart])

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play().catch(() => {})
    } else {
      video.pause()
    }
  }, [])

  const toggleMute = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    video.muted = !video.muted
  }, [])

  const setVolume = useCallback((value: number) => {
    const clamped = Math.min(1, Math.max(0, value))
    const video = videoRef.current
    if (video) {
      video.volume = clamped
      // Raising the slider above 0 should audibly take effect even if the
      // user had previously muted — otherwise "Volume" silently does nothing.
      if (clamped > 0 && video.muted) video.muted = false
    }
    setVolumeState(clamped)
  }, [])

  // Mirrors the <video> element's own play/pause/volume state rather than
  // tracking it independently, so external changes (native keyboard
  // shortcuts, the loop restarting playback, etc.) stay reflected.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleVolumeChange = () => {
      setIsMuted(video.muted)
      setVolumeState(video.volume)
    }

    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('volumechange', handleVolumeChange)

    return () => {
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('volumechange', handleVolumeChange)
    }
  }, [])

  return { videoRef, hasStarted, isPlaying, isMuted, volume, start, togglePlay, restart, toggleMute, setVolume }
}

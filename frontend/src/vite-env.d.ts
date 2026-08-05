/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DETECTION_WS_URL?: string
  readonly VITE_USE_MOCK_DATA?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

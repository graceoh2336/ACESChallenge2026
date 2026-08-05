export function CameraFeedPlaceholder() {
  return (
    <div className="absolute inset-0 overflow-hidden">
      {/* Night sky */}
      <div className="absolute inset-0 bg-[linear-gradient(180deg,#04050a_0%,#0a0e18_38%,#141a28_58%,#1c2131_100%)]" />

      {/* Distant city glow */}
      <div className="absolute inset-x-0 top-[30%] h-40 bg-[radial-gradient(60%_100%_at_20%_100%,rgba(255,176,32,0.12),transparent),radial-gradient(40%_100%_at_75%_100%,rgba(59,167,255,0.1),transparent)] blur-xl" />

      {/* Road surface */}
      <div
        className="absolute bottom-0 left-1/2 h-[58%] w-[130%] -translate-x-1/2 bg-[linear-gradient(180deg,#1c1f28_0%,#101319_60%,#0a0c11_100%)]"
        style={{ clipPath: 'polygon(38% 0%, 62% 0%, 100% 100%, 0% 100%)' }}
      >
        {/* Centre lane dashes */}
        <div className="absolute left-1/2 top-0 h-full w-[3%] -translate-x-1/2 overflow-hidden">
          <div className="h-[200%] w-full animate-scan bg-[repeating-linear-gradient(180deg,rgba(230,200,120,0.55)_0px,rgba(230,200,120,0.55)_26px,transparent_26px,transparent_58px)]" />
        </div>
        {/* Edge lines */}
        <div className="absolute bottom-0 left-[6%] top-0 w-[2px] bg-cockpit-100/25" />
        <div className="absolute bottom-0 right-[6%] top-0 w-[2px] bg-cockpit-100/25" />
      </div>

      {/* Ambient bokeh lights */}
      <div className="absolute left-[18%] top-[42%] h-3 w-3 rounded-full bg-status-amber/70 blur-[3px]" />
      <div className="absolute right-[22%] top-[46%] h-2 w-2 rounded-full bg-status-red/70 blur-[2px]" />
      <div className="absolute right-[38%] top-[38%] h-2.5 w-2.5 rounded-full bg-status-blue/60 blur-[3px]" />

      {/* HUD grid + vignette */}
      <div className="grid-overlay absolute inset-0 opacity-[0.08]" />
      <div className="absolute inset-0 shadow-[inset_0_0_120px_60px_rgba(0,0,0,0.65)]" />
    </div>
  )
}

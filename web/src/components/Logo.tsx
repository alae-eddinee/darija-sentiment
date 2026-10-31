export default function Logo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <div className={`relative flex items-center justify-center shrink-0 ${className}`}>
      <svg
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
      >
        {/* Minimal Obsidian Tile */}
        <rect width="36" height="36" rx="9" fill="#09090b" />
        <rect
          x="0.5"
          y="0.5"
          width="35"
          height="35"
          rx="8.5"
          stroke="rgba(255, 255, 255, 0.12)"
        />

        {/* Clean Moroccan Pentagram Geometry */}
        <path
          d="M 18,7.5 L 24.16,26.47 L 8.04,14.76 L 27.96,14.76 L 11.84,26.47 Z"
          stroke="#ffffff"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </svg>
    </div>
  );
}

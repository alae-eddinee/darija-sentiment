export default function Logo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <div className={`relative flex items-center justify-center shrink-0 ${className}`}>
      <svg
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full drop-shadow-sm"
      >
        {/* Obsidian Base Tile */}
        <rect width="40" height="40" rx="10" fill="#09090b" />
        <rect
          x="0.5"
          y="0.5"
          width="39"
          height="39"
          rx="9.5"
          stroke="rgba(255,255,255,0.18)"
        />

        {/* Moroccan Geometric Pentagram Star */}
        <path
          d="M20 9.5 L23.09 18.91 L33.09 18.91 L25 24.79 L28.09 34.21 L20 28.33 L11.91 34.21 L15 24.79 L6.91 18.91 L16.91 18.91 Z"
          stroke="#ffffff"
          strokeWidth="2.2"
          strokeLinejoin="round"
          strokeLinecap="round"
          fill="none"
        />
        {/* Subtle center jewel */}
        <circle cx="20" cy="21.5" r="2.2" fill="#22c55e" />
      </svg>
    </div>
  );
}

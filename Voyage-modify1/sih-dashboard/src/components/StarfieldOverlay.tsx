import React, { useMemo } from 'react';

export type StarDimmerMode = 'cinematic' | 'deep' | 'subtle' | 'off';

interface StarfieldOverlayProps {
  dimmerMode?: StarDimmerMode;
  showStars?: boolean;
  className?: string;
}

export const StarfieldOverlay: React.FC<StarfieldOverlayProps> = ({
  dimmerMode = 'cinematic',
  showStars = true,
  className = '',
}) => {
  // Atmospheric dimming vignette
  const vignetteStyle = useMemo(() => {
    switch (dimmerMode) {
      case 'deep':
        return 'radial-gradient(circle at 50% 50%, rgba(3, 4, 7, 0) 15%, rgba(4, 5, 8, 0.45) 50%, rgba(3, 4, 6, 0.85) 75%, rgba(2, 3, 5, 0.95) 100%)';
      case 'cinematic':
        return 'radial-gradient(circle at 50% 50%, rgba(3, 4, 7, 0) 22%, rgba(5, 7, 10, 0.32) 55%, rgba(4, 5, 8, 0.65) 80%, rgba(2, 3, 5, 0.88) 100%)';
      case 'subtle':
        return 'radial-gradient(circle at 50% 50%, rgba(3, 4, 7, 0) 35%, rgba(6, 8, 12, 0.20) 65%, rgba(4, 5, 8, 0.50) 100%)';
      case 'off':
      default:
        return 'none';
    }
  }, [dimmerMode]);

  return (
    <div
      className={`absolute inset-0 pointer-events-none overflow-hidden select-none bg-[#030406] ${className}`}
      aria-hidden="true"
    >
      {/* ── 1. AUTHENTIC PHOTOGRAPHIC MILKY WAY STARFIELD (Natural Deep Space) ── */}
      {showStars && (
        <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none">
          <img
            src="/assets/real_milkyway_starfield.png"
            alt="Authentic Astrophotography Milky Way Starfield"
            className="w-full h-full object-cover pointer-events-none select-none animate-real-starfield"
          />
        </div>
      )}

      {/* ── 2. ATMOSPHERIC DIMMING VIGNETTE LAYER (Earth-like sky darkening) ── */}
      {dimmerMode !== 'off' && (
        <div
          className="absolute inset-0 w-full h-full transition-opacity duration-700 pointer-events-none"
          style={{ background: vignetteStyle }}
        />
      )}
    </div>
  );
};

export default StarfieldOverlay;

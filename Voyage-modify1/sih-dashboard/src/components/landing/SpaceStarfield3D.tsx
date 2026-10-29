import React, { useEffect, useRef } from 'react';

export const SpaceStarfield3D: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || window.innerWidth);
    let height = (canvas.height = canvas.parentElement?.clientHeight || window.innerHeight);

    // Generate 3D Starfield Particles with Depth (Z)
    const starCount = 320;
    const stars = Array.from({ length: starCount }, () => ({
      x: (Math.random() - 0.5) * width * 1.8,
      y: (Math.random() - 0.5) * height * 1.8,
      z: Math.random() * 1000 + 10,
      radius: Math.random() * 1.4 + 0.3,
      alpha: Math.random() * 0.75 + 0.25,
      hue: Math.random() > 0.85 ? 42 : Math.random() > 0.7 ? 205 : 0, // Warm Gold, Lunar Cyan, Silver White
      twinkleSpeed: Math.random() * 0.02 + 0.005,
    }));

    let mouseX = 0;
    let mouseY = 0;
    let targetMouseX = 0;
    let targetMouseY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      targetMouseX = (e.clientX / window.innerWidth - 0.5) * 60;
      targetMouseY = (e.clientY / window.innerHeight - 0.5) * 60;
    };

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.parentElement?.clientWidth || window.innerWidth;
      height = canvas.height = canvas.parentElement?.clientHeight || window.innerHeight;
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    window.addEventListener('resize', handleResize);

    const render = () => {
      // Smooth mouse parallax damping
      mouseX += (targetMouseX - mouseX) * 0.05;
      mouseY += (targetMouseY - mouseY) * 0.05;

      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;

      for (let i = 0; i < stars.length; i++) {
        const star = stars[i];

        // Slowly drift stars toward viewer
        star.z -= 0.35;
        if (star.z <= 0) {
          star.z = 1000;
          star.x = (Math.random() - 0.5) * width * 1.8;
          star.y = (Math.random() - 0.5) * height * 1.8;
        }

        // Perspective 3D projection
        const k = 450 / star.z;
        const px = (star.x + mouseX * (1 - star.z / 1000)) * k + cx;
        const py = (star.y + mouseY * (1 - star.z / 1000)) * k + cy;

        if (px >= 0 && px < width && py >= 0 && py < height) {
          // Dynamic twinkle
          star.alpha += Math.sin(Date.now() * star.twinkleSpeed) * 0.008;
          const currentAlpha = Math.max(0.15, Math.min(0.95, star.alpha));
          const size = Math.max(0.4, star.radius * k * 0.8);

          ctx.beginPath();
          ctx.arc(px, py, size, 0, Math.PI * 2);

          if (star.hue === 42) {
            ctx.fillStyle = `rgba(214, 195, 139, ${currentAlpha})`; // Champagne Gold
            ctx.shadowColor = 'rgba(214, 195, 139, 0.6)';
            ctx.shadowBlur = size * 2;
          } else if (star.hue === 205) {
            ctx.fillStyle = `rgba(147, 197, 253, ${currentAlpha})`; // Cyan
            ctx.shadowColor = 'rgba(147, 197, 253, 0.5)';
            ctx.shadowBlur = size * 2;
          } else {
            ctx.fillStyle = `rgba(240, 244, 248, ${currentAlpha})`; // Brilliant Silver White
            ctx.shadowBlur = 0;
          }

          ctx.fill();
        }
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none z-0 opacity-75"
      style={{ mixBlendMode: 'screen' }}
    />
  );
};

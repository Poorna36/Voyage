import { useEffect } from 'react';

/**
 * useMagneticButtons — High-performance 3D magnetic spring interaction
 * 
 * Attaches delegated pointer physics across all interactive buttons, pills,
 * and controls. Provides smooth sub-pixel magnetic attraction with spring damping
 * and dynamic light reflection coordinates.
 */
export function useMagneticButtons() {
  useEffect(() => {
    // Only enable magnetic effects on devices with fine pointer (mouse/trackpad)
    if (window.matchMedia('(pointer: coarse)').matches) return;

    let activeElement: HTMLElement | null = null;
    let currentX = 0;
    let currentY = 0;
    let targetX = 0;
    let targetY = 0;
    let targetRotX = 0;
    let targetRotY = 0;
    let currentRotX = 0;
    let currentRotY = 0;
    let strength = 0.28;
    let maxDist = 12;
    let animFrame: number | null = null;
    let isRunning = false;

    const springLoop = () => {
      // Lerp translation
      currentX += (targetX - currentX) * 0.18;
      currentY += (targetY - currentY) * 0.18;
      // Lerp 3D tilt
      currentRotX += (targetRotX - currentRotX) * 0.18;
      currentRotY += (targetRotY - currentRotY) * 0.18;

      if (activeElement) {
        activeElement.style.transform = `translate3d(${currentX.toFixed(2)}px, ${currentY.toFixed(2)}px, 0px) rotateX(${currentRotX.toFixed(2)}deg) rotateY(${currentRotY.toFixed(2)}deg)`;
      }

      // Continue loop if active or still returning to rest
      const isSettled =
        Math.abs(targetX - currentX) < 0.05 &&
        Math.abs(targetY - currentY) < 0.05 &&
        Math.abs(targetRotX - currentRotX) < 0.05 &&
        Math.abs(targetRotY - currentRotY) < 0.05;

      if (!isSettled || activeElement !== null) {
        animFrame = requestAnimationFrame(springLoop);
      } else {
        isRunning = false;
      }
    };


    const handlePointerMove = (e: PointerEvent) => {
      const target = (e.target as HTMLElement)?.closest(
        'button, [data-magnetic="true"], [role="button"], .magnetic-btn'
      ) as HTMLElement | null;

      // Ignore disabled or explicitly non-magnetic elements
      if (!target || target.hasAttribute('disabled') || target.getAttribute('data-magnetic') === 'false') {
        if (activeElement) {
          targetX = 0;
          targetY = 0;
          targetRotX = 0;
          targetRotY = 0;
          activeElement = null;
        }
        return;
      }

      if (activeElement !== target) {
        if (activeElement) {
          activeElement.style.transform = '';
        }
        activeElement = target;
        const customStrength = target.getAttribute('data-magnetic-strength');
        strength = customStrength ? parseFloat(customStrength) : 0.28;
        maxDist = target.offsetWidth > 150 ? 14 : 9;
      }

      const rect = target.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      const rawDeltaX = (e.clientX - centerX) * strength;
      const rawDeltaY = (e.clientY - centerY) * strength;

      targetX = Math.max(-maxDist, Math.min(maxDist, rawDeltaX));
      targetY = Math.max(-maxDist, Math.min(maxDist, rawDeltaY));

      // Subtle 3D tilt towards mouse position
      targetRotX = Math.max(-6, Math.min(6, -rawDeltaY * 0.4));
      targetRotY = Math.max(-6, Math.min(6, rawDeltaX * 0.4));

      // Dynamic radial light highlight tracking
      const mouseRelX = ((e.clientX - rect.left) / rect.width) * 100;
      const mouseRelY = ((e.clientY - rect.top) / rect.height) * 100;
      target.style.setProperty('--mouse-x', `${mouseRelX.toFixed(1)}%`);
      target.style.setProperty('--mouse-y', `${mouseRelY.toFixed(1)}%`);

      if (!isRunning) {
        isRunning = true;
        animFrame = requestAnimationFrame(springLoop);
      }
    };

    const handlePointerLeave = () => {
      if (activeElement) {
        activeElement.style.transform = '';
      }
      targetX = 0;
      targetY = 0;
      targetRotX = 0;
      targetRotY = 0;
      activeElement = null;
    };


    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    document.addEventListener('pointerleave', handlePointerLeave, { passive: true });

    return () => {
      if (animFrame) cancelAnimationFrame(animFrame);
      window.removeEventListener('pointermove', handlePointerMove);
      document.removeEventListener('pointerleave', handlePointerLeave);
    };
  }, []);
}

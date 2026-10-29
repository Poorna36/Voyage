import React, { useState, useRef, useEffect } from 'react';
import { Camera, Radio, Sun, Compass, Layers, X, Sparkles, RotateCcw } from 'lucide-react';
import { chandrayaanTransparentImg } from '../../data/lunarisDatasets';

interface ChandrayaanViewerProps {
  onLaunchWorkbench?: () => void;
}

interface SensorCallout {
  id: string;
  name: string;
  sub: string;
  specs: string;
  desc: string;
  targetX: number; // Point on satellite (%)
  targetY: number; // Point on satellite (%)
  labelX: number;  // Callout box position (%)
  labelY: number;  // Callout box position (%)
  side: 'left' | 'right' | 'top' | 'bottom';
  icon: 'camera' | 'radio' | 'sun' | 'sensor' | 'propulsion';
}

const SENSORS: SensorCallout[] = [
  {
    id: 'hga',
    name: 'HIGH GAIN ANTENNA',
    sub: 'Parabolic Dish Reflector',
    specs: 'Dual-Axis Steerable • X-Band',
    desc: 'Downlinks high-speed orbital imagery to ISRO Deep Space Network Byalalu.',
    targetX: 24,
    targetY: 42,
    labelX: 5,
    labelY: 18,
    side: 'left',
    icon: 'radio'
  },
  {
    id: 'ohrc',
    name: 'OHRC',
    sub: 'Orbiter High Resolution Camera',
    specs: '0.25 m / px • 450–900 nm',
    desc: 'Nadir-pointing panchromatic optical telescope for micro-crater hazard mapping.',
    targetX: 23,
    targetY: 60,
    labelX: 5,
    labelY: 68,
    side: 'left',
    icon: 'camera'
  },
  {
    id: 'tmc2',
    name: 'TMC-2',
    sub: 'Terrain Mapping Camera-2',
    specs: '5.0 m / px • Triplet Stereo',
    desc: 'Fore (+26°), nadir (0°), and aft (-26°) stereo lenses generating 3D lunar DEMs.',
    targetX: 41,
    targetY: 33,
    labelX: 30,
    labelY: 7,
    side: 'top',
    icon: 'camera'
  },
  {
    id: 'iirs',
    name: 'IIRS',
    sub: 'Imaging IR Spectrometer',
    specs: '250 bands • 800 – 5000 nm',
    desc: 'Detects diagnostic 1.5 µm & 2.0 µm molecular absorption lines of lunar water-ice.',
    targetX: 33,
    targetY: 57,
    labelX: 18,
    labelY: 88,
    side: 'bottom',
    icon: 'sensor'
  },
  {
    id: 'solar',
    name: 'SOLAR ARRAY WING',
    sub: 'Photovoltaic Wing',
    specs: '1000 W Output • Single-Wing',
    desc: 'Multi-junction high-efficiency solar panels tracking the sun for payload power.',
    targetX: 74,
    targetY: 46,
    labelX: 72,
    labelY: 22,
    side: 'right',
    icon: 'sun'
  },
  {
    id: 'lam',
    name: 'LAM & RCS THRUSTERS',
    sub: 'Liquid Apogee Motor',
    specs: '440 N Main Engine + 8 RCS',
    desc: 'Provides orbit circularization and 3-axis autonomous attitude stabilization.',
    targetX: 51,
    targetY: 76,
    labelX: 68,
    labelY: 84,
    side: 'right',
    icon: 'propulsion'
  }
];

export const ChandrayaanViewer: React.FC<ChandrayaanViewerProps> = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // 3D Model rotation state
  const rotRef = useRef({ yaw: 0, pitch: 0, velYaw: 0, velPitch: 0 });
  const [rotation, setRotation] = useState({ yaw: 0, pitch: 0 });
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const hasMovedRef = useRef(false);

  const [showSensors, setShowSensors] = useState(false);
  const [selectedSensor, setSelectedSensor] = useState<string | null>(null);

  // 3D Physics & Angular Momentum Animation Loop
  useEffect(() => {
    let animId: number;

    const updatePhysics = () => {
      if (!isDraggingRef.current) {
        if (!showSensors) {
          // Subtle zero-gravity orbital drift
          rotRef.current.velYaw += (0.04 - rotRef.current.velYaw) * 0.03;
          rotRef.current.velPitch *= 0.94;
          rotRef.current.yaw += rotRef.current.velYaw;
          rotRef.current.pitch += rotRef.current.velPitch;

          // Limit pitch to natural bounds
          rotRef.current.pitch = Math.max(-28, Math.min(28, rotRef.current.pitch));

          setRotation({
            yaw: rotRef.current.yaw,
            pitch: rotRef.current.pitch
          });
        } else {
          // Smooth return to center when sensors are shown
          rotRef.current.yaw *= 0.88;
          rotRef.current.pitch *= 0.88;
          setRotation({
            yaw: rotRef.current.yaw,
            pitch: rotRef.current.pitch
          });
        }
      }

      animId = requestAnimationFrame(updatePhysics);
    };

    animId = requestAnimationFrame(updatePhysics);
    return () => cancelAnimationFrame(animId);
  }, [showSensors]);

  // Pointer Drag Interaction
  const handlePointerDown = (e: React.PointerEvent) => {
    isDraggingRef.current = true;
    hasMovedRef.current = false;
    dragStartRef.current = { x: e.clientX, y: e.clientY };
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;

    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      hasMovedRef.current = true;
    }

    dragStartRef.current = { x: e.clientX, y: e.clientY };

    const sensitivity = 0.35;
    rotRef.current.yaw += dx * sensitivity;
    rotRef.current.pitch -= dy * sensitivity;
    rotRef.current.pitch = Math.max(-35, Math.min(35, rotRef.current.pitch));

    rotRef.current.velYaw = dx * 0.15;
    rotRef.current.velPitch = -dy * 0.15;

    setRotation({
      yaw: rotRef.current.yaw,
      pitch: rotRef.current.pitch
    });
  };

  const handlePointerUp = () => {
    isDraggingRef.current = false;
  };

  // Toggle sensors when clicked without dragging
  const handleContainerClick = () => {
    if (hasMovedRef.current) return; // Ignore if user was rotating/dragging
    setShowSensors(!showSensors);
    if (showSensors) {
      setSelectedSensor(null);
    }
  };

  const handleReset = (e: React.MouseEvent) => {
    e.stopPropagation();
    rotRef.current = { yaw: 0, pitch: 0, velYaw: 0, velPitch: 0 };
    setRotation({ yaw: 0, pitch: 0 });
    setShowSensors(false);
    setSelectedSensor(null);
  };

  // Calculate dynamic solar specular glare based on 3D rotation
  const sunX = Math.max(10, Math.min(90, 45 + (rotation.yaw % 360) * 0.8));
  const sunY = Math.max(10, Math.min(90, 40 - rotation.pitch * 0.9));

  return (
    <div
      ref={containerRef}
      className="relative w-full max-w-[440px] sm:max-w-[480px] md:max-w-[520px] aspect-square flex items-center justify-center select-none"
    >
      {/* Background Orbit Ring Vectors in Deep Space */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none opacity-35 animate-spin-slow"
        viewBox="0 0 500 500"
        style={{ animationDuration: '140s' }}
      >
        <circle cx="250" cy="250" r="235" fill="none" stroke="rgba(214, 195, 139, 0.12)" strokeDasharray="3 6" />
        <circle cx="250" cy="250" r="195" fill="none" stroke="rgba(214, 195, 139, 0.16)" strokeWidth="1" />
        <ellipse
          cx="250"
          cy="250"
          rx="225"
          ry="90"
          transform="rotate(-26 250 250)"
          fill="none"
          stroke="#D6C38B"
          strokeWidth="1.2"
          strokeDasharray="4 4"
          className="opacity-45"
        />
      </svg>

      {/* 3D Interactive Chandrayaan-2 Spacecraft Model Container */}
      <div
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onClick={handleContainerClick}
        style={{
          transform: `perspective(1200px) rotateX(${rotation.pitch}deg) rotateY(${rotation.yaw}deg)`,
          transformStyle: 'preserve-3d',
          transition: isDraggingRef.current ? 'none' : 'transform 0.08s ease-out'
        }}
        className="relative w-full h-full flex items-center justify-center cursor-grab active:cursor-grabbing touch-none"
      >
        {/* Actual Hyperrealistic 8K Chandrayaan-2 Spacecraft Asset (No loss in realism) */}
        <img
          src={chandrayaanTransparentImg}
          alt="ISRO Chandrayaan-2 Spacecraft 3D Model"
          className="w-full h-full object-contain filter drop-shadow-[0_15px_35px_rgba(214,195,139,0.22)] pointer-events-none"
          draggable={false}
        />

        {/* Dynamic Specular Sunlight Sweep Layer across Gold Foil and Solar Array */}
        <div
          style={{
            background: `radial-gradient(circle 140px at ${sunX}% ${sunY}%, rgba(255, 235, 175, 0.28) 0%, rgba(214, 195, 139, 0.08) 50%, transparent 80%)`,
            mixBlendMode: 'screen'
          }}
          className="absolute inset-0 rounded-full pointer-events-none transition-all duration-75"
        />

        {/* SVG LEADER LINES OVERLAY (Appears on click when showSensors is active) */}
        {showSensors && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-20" viewBox="0 0 100 100">
            {SENSORS.map((s) => {
              const isSelected = selectedSensor === s.id;
              const strokeColor = isSelected ? '#FAF6EB' : '#D6C38B';
              const strokeWidth = isSelected ? 0.6 : 0.35;

              return (
                <g key={`line-${s.id}`} className="animate-in fade-in duration-300">
                  <line
                    x1={s.labelX}
                    y1={s.labelY}
                    x2={s.targetX}
                    y2={s.targetY}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeDasharray={isSelected ? 'none' : '1 1'}
                    className="opacity-75"
                  />
                  <circle
                    cx={s.targetX}
                    cy={s.targetY}
                    r={isSelected ? 1.4 : 0.9}
                    fill={strokeColor}
                    stroke="#000"
                    strokeWidth={0.3}
                  />
                </g>
              );
            })}
          </svg>
        )}

        {/* SENSOR PINPOINT LABELS & TEXT OVERLAY */}
        {showSensors && (
          <div className="absolute inset-0 z-30 pointer-events-auto">
            {SENSORS.map((s) => {
              const isSelected = selectedSensor === s.id;

              return (
                <div
                  key={s.id}
                  style={{ left: `${s.labelX}%`, top: `${s.labelY}%` }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedSensor(isSelected ? null : s.id);
                  }}
                  className={`absolute -translate-x-1/2 -translate-y-1/2 p-2 rounded-xl border backdrop-blur-xl transition-all duration-200 cursor-pointer shadow-2xl max-w-[150px] sm:max-w-[170px] ${
                    isSelected
                      ? 'bg-[#0D121B]/95 border-[#FAF6EB] shadow-[0_0_20px_rgba(214,195,139,0.35)] scale-105 z-40'
                      : 'bg-[#070A0F]/90 border-[#D6C38B]/50 hover:border-[#D6C38B] hover:bg-[#0D121B]/95 z-30'
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-mono-tech text-[9.5px] font-bold text-[#D6C38B] uppercase tracking-wider">
                    {s.icon === 'camera' && <Camera size={11} />}
                    {s.icon === 'radio' && <Radio size={11} />}
                    {s.icon === 'sun' && <Sun size={11} />}
                    {s.icon === 'sensor' && <Compass size={11} />}
                    {s.icon === 'propulsion' && <Layers size={11} />}
                    <span className="truncate">{s.name}</span>
                  </div>

                  <div className="font-mono-tech text-[8.5px] text-[#FAF6EB]/90 font-medium pt-0.5">
                    {s.specs}
                  </div>

                  <div className="font-sans text-[8px] text-slate-300 leading-tight pt-1 border-t border-subtle/50 mt-1">
                    {s.desc}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Interactive Controls & Sensor Toggle Pill */}
      <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2">
        <button
          onClick={() => setShowSensors(!showSensors)}
          className={`flex items-center gap-1.5 px-3 py-1 rounded-full border font-mono-tech text-[9.5px] tracking-wider uppercase transition-all duration-200 cursor-pointer shadow-lg ${
            showSensors
              ? 'bg-[#D6C38B] text-black border-[#D6C38B] font-bold shadow-[0_0_15px_rgba(214,195,139,0.4)]'
              : 'bg-[#07090C]/85 hover:bg-[#0E131A] text-[#D6C38B] hover:text-[#FAF6EB] border-[#D6C38B]/40 hover:border-[#D6C38B]'
          }`}
        >
          {showSensors ? <X size={11} /> : <Sparkles size={11} />}
          <span>{showSensors ? 'CLOSE SENSORS' : 'INSPECT SENSORS'}</span>
        </button>

        {(rotation.yaw !== 0 || rotation.pitch !== 0) && (
          <button
            onClick={handleReset}
            title="Reset 3D Orientation"
            className="p-1 rounded-full bg-[#07090C]/85 hover:bg-[#0E131A] border border-[#D6C38B]/40 hover:border-[#D6C38B] text-[#D6C38B] hover:text-[#FAF6EB] transition-colors cursor-pointer"
          >
            <RotateCcw size={11} />
          </button>
        )}
      </div>
    </div>
  );
};

export default ChandrayaanViewer;

import React from 'react';

// Monochrome Shield Icon for Resolution control
export const ShieldIcon: React.FC<{ className?: string; size?: number }> = ({ className = '', size = 20 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    className={className}
    aria-hidden="true"
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M12 2.25c-.29 0-.57.14-.74.37L3.92 11.9a1.25 1.25 0 0 0-.17.65c0 6.64 4.5 10.3 8.25 11.2 3.75-.9 8.25-4.56 8.25-11.2 0-.23-.06-.46-.17-.65L12.74 2.62a1 1 0 0 0-.74-.37zm0 2.25 6.25 7.5c0 4.88-3.3 7.82-6.25 8.68-2.95-.86-6.25-3.8-6.25-8.68L12 4.5z"
    />
  </svg>
);

// Monochrome User/Silhouette Icon for Format control
export const UserIcon: React.FC<{ className?: string; size?: number }> = ({ className = '', size = 20 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    className={className}
    aria-hidden="true"
  >
    <circle cx="12" cy="7" r="4.2" />
    <path d="M12 13.5c-4.4 0-8 2.6-8 6.2 0 .5.4.8.9.8h14.2c.5 0 .9-.3.9-.8 0-3.6-3.6-6.2-8-6.2z" />
  </svg>
);

// Chevron Down Icon for Dropdowns
export const ChevronDownIcon: React.FC<{ className?: string; size?: number }> = ({ className = '', size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.4"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

// Circle Plus Icon for Lower Sidebar placeholder options
export const PlusCircleIcon: React.FC<{ className?: string; size?: number }> = ({ className = '', size = 20 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.7"
    className={className}
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="9.5" />
    <line x1="12" y1="8" x2="12" y2="16" strokeLinecap="round" strokeWidth="1.8" />
    <line x1="8" y1="12" x2="16" y2="12" strokeLinecap="round" strokeWidth="1.8" />
  </svg>
);

// Window Control: Minimize (horizontal bar)
export const MinimizeIcon: React.FC<{ size?: number }> = ({ size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <rect x="2" y="7.5" width="12" height="1.6" rx="0.8" />
  </svg>
);

// Window Control: Maximize (square outline)
export const MaximizeIcon: React.FC<{ size?: number }> = ({ size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
    <rect x="2.5" y="2.5" width="11" height="11" rx="1.5" />
  </svg>
);

// Window Control: Restore
export const RestoreIcon: React.FC<{ size?: number }> = ({ size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
    <rect x="4.5" y="1.5" width="9" height="9" rx="1.2" />
    <polyline points="1.5,5.5 1.5,14.5 10.5,14.5" />
  </svg>
);

// Window Control: Close (cross)
export const CloseIcon: React.FC<{ size?: number }> = ({ size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true">
    <line x1="3" y1="3" x2="13" y2="13" />
    <line x1="13" y1="3" x2="3" y2="13" />
  </svg>
);

// Apple / iOS Settings Gear Icon (Squircle container with gear geometry)
export const AppleSettingsGearIcon: React.FC<{ size?: number; className?: string }> = ({ size = 48, className = '' }) => {
  // Generate outer gear teeth around perimeter
  const teethCount = 18;
  const teeth = [];
  for (let i = 0; i < teethCount; i++) {
    const angle = (i * 360) / teethCount;
    teeth.push(
      <rect
        key={i}
        x="28.8"
        y="6"
        width="6.4"
        height="7.5"
        rx="1.2"
        fill="#5a5a5e"
        transform={`rotate(${angle} 32 32)`}
      />
    );
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      className={className}
      aria-hidden="true"
    >
      <defs>
        {/* iOS-style metallic squircle gradient */}
        <linearGradient id="iosSquircleBg" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#4a4a4d" />
          <stop offset="50%" stopColor="#363638" />
          <stop offset="100%" stopColor="#252527" />
        </linearGradient>
        
        {/* Inner concentric ring metallic gradient */}
        <radialGradient id="gearCenterGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#68686d" />
          <stop offset="65%" stopColor="#3c3c3f" />
          <stop offset="100%" stopColor="#28282a" />
        </radialGradient>

        {/* Squircle outer drop-shadow / bevel */}
        <filter id="gearBevel" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="1.5" stdDeviation="1" floodColor="#000000" floodOpacity="0.35" />
        </filter>
      </defs>

      {/* Squircle icon base */}
      <rect
        x="3"
        y="3"
        width="58"
        height="58"
        rx="14"
        fill="url(#iosSquircleBg)"
        stroke="#636368"
        strokeWidth="1.2"
        filter="url(#gearBevel)"
      />

      {/* Subtle top edge highlight */}
      <path
        d="M17 4.2 h30 c7 0 12 5 12 12"
        stroke="#808085"
        strokeWidth="0.8"
        strokeLinecap="round"
        opacity="0.4"
      />

      {/* Group: Concentric Gear Mechanism */}
      <g id="gear-assembly">
        {/* Outer teeth */}
        {teeth}

        {/* Outer gear rim */}
        <circle cx="32" cy="32" r="21" fill="#444447" stroke="#252528" strokeWidth="1" />
        <circle cx="32" cy="32" r="18" fill="url(#gearCenterGrad)" stroke="#55555a" strokeWidth="0.8" />

        {/* Concentric etched grooves */}
        <circle cx="32" cy="32" r="14.5" fill="none" stroke="#222224" strokeWidth="1.4" />
        <circle cx="32" cy="32" r="11" fill="none" stroke="#636368" strokeWidth="0.8" opacity="0.6" />

        {/* Inner central hub */}
        <circle cx="32" cy="32" r="7.5" fill="#1e1e20" stroke="#707075" strokeWidth="1.2" />
        <circle cx="32" cy="32" r="3.2" fill="#58585e" />
      </g>
    </svg>
  );
};

// Center No-AI Prohibition Symbol:
// Circle with diagonal slash across the stylized AI geometric neural knot
export const NoAISymbol: React.FC<{ size?: number; className?: string }> = ({ size, className = '' }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      fill="none"
      className={className}
      aria-hidden="true"
    >
      <defs>
        {/* Prohibition mark clip path to keep clean geometry */}
        <clipPath id="circleClip">
          <circle cx="100" cy="100" r="92" />
        </clipPath>
      </defs>

      {/* Outer Prohibition Circle Outline */}
      <circle
        cx="100"
        cy="100"
        r="91"
        stroke="#424246"
        strokeWidth="9"
        fill="none"
      />

      {/* Nested Stylized AI / Neural Knot (OpenAI-style 6-fold looping knot) */}
      <g
        transform="translate(100, 100) scale(1.15) translate(-50, -50)"
        stroke="#424246"
        strokeWidth="6.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      >
        {/* 6 interconnected symmetric loop arms of the generative AI emblem */}
        {/* Loop 1: Top Right */}
        <path d="M 50 18 C 66 18 78 28 78 44 C 78 54 72 63 62 68 L 50 75" />
        <path d="M 50 75 L 50 48" />

        {/* Loop 2: Right */}
        <path d="M 77 34 C 85 48 83 65 72 75 C 63 83 52 84 41 80 L 29 74" />
        <path d="M 29 74 L 52 61" />

        {/* Loop 3: Bottom Right */}
        <path d="M 73 66 C 68 81 53 90 38 87 C 27 85 19 77 17 66 L 16 52" />
        <path d="M 16 52 L 39 52" />

        {/* Loop 4: Bottom Left */}
        <path d="M 50 82 C 34 82 22 72 22 56 C 22 46 28 37 38 32 L 50 25" />
        <path d="M 50 25 L 50 52" />

        {/* Loop 5: Left */}
        <path d="M 23 66 C 15 52 17 35 28 25 C 37 17 48 16 59 20 L 71 26" />
        <path d="M 71 26 L 48 39" />

        {/* Loop 6: Top Left */}
        <path d="M 27 34 C 32 19 47 10 62 13 C 73 15 81 23 83 34 L 84 48" />
        <path d="M 84 48 L 61 48" />

        {/* Inner center core rings */}
        <circle cx="50" cy="50" r="9" stroke="#424246" strokeWidth="5.5" />
      </g>

      {/* The Diagonal Prohibition Strike-through Line */}
      {/* Runs from top-left (approx 135deg) to bottom-right (approx 315deg) */}
      <line
        x1="36"
        y1="36"
        x2="164"
        y2="164"
        stroke="#424246"
        strokeWidth="9"
        strokeLinecap="round"
      />
    </svg>
  );
};

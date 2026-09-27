import React from 'react';

export const Branding: React.FC = () => {
  return (
    <div className="branding-section">
      {/* Primary Brand Logo Emblem */}
      <div className="sidebar-brand-card" role="banner" aria-label="Concealed brand emblem">
        <div className="sidebar-brand-emblem-wrap">
          <div className="brand-emblem-glow" aria-hidden="true" />
          <img
            src="/logo_white_cropped.png"
            alt="Concealed Logo - Obfuscated Face Silhouette"
            className="sidebar-brand-logo-img"
          />
          <div className="brand-emblem-ring" aria-hidden="true" />
        </div>
        <div className="sidebar-brand-text-lockup">
          <span className="sidebar-brand-title">CONCEALED</span>
          <span className="sidebar-brand-tagline">AI VISION DEFENSE</span>
        </div>
      </div>

      <p className="branding-subtitle">
        STAY SAFE. NOT SORRY.<br />
        <span>KEEP YOUR IMAGE PRIVATE FROM AI</span>
      </p>
      <div className="branding-divider" role="separator" aria-hidden="true" />
    </div>
  );
};

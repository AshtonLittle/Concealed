import React from 'react';

export const Branding: React.FC = () => {
  return (
    <div className="branding-section">
      {/* Top Left Logo Placeholder replacing Image Concealer text */}
      <div className="sidebar-logo-placeholder" role="img" aria-label="Logo placeholder">
        <div className="sidebar-logo-frame">
          <svg
            className="sidebar-logo-icon"
            viewBox="0 0 24 24"
            width="22"
            height="22"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <rect x="3" y="3" width="18" height="18" rx="4" strokeDasharray="3 3" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
          <span className="sidebar-logo-text">LOGO PLACEHOLDER</span>
        </div>
      </div>

      <p className="branding-subtitle">
        KEEP YOUR IMAGE<br />
        PRIVATE FROM AI
      </p>
      <div className="branding-divider" role="separator" aria-hidden="true" />
    </div>
  );
};

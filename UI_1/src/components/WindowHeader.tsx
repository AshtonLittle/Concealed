import React from 'react';

export const WindowHeader: React.FC = () => {
  return (
    <header className="window-header" aria-label="Application title bar">
      {/* Left side: Classic CONCEALED wordmark leading the top bar & PORTABLE tab */}
      <div className="topbar-left" role="presentation">
        <span className="concealed-classic-title">CONCEALED</span>
        <div className="portable-tab">
          <span className="portable-text">PORTABLE</span>
        </div>
      </div>

      {/* Right side has no window controls (X, minimize, maximize removed) */}
      <div className="topbar-right" aria-hidden="true" />
    </header>
  );
};

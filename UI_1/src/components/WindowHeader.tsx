import React from 'react';

export type AppView = 'image' | 'video' | 'benchmark';

interface WindowHeaderProps {
  currentView?: AppView;
  onNavigate?: (view: AppView) => void;
}

export const WindowHeader: React.FC<WindowHeaderProps> = ({
  currentView = 'image',
  onNavigate,
}) => {
  return (
    <header className="window-header" aria-label="Application title bar">
      <div className="topbar-left" role="presentation">
        {/* Classic CONCEALED wordmark with Logo */}
        <button
          type="button"
          className="concealed-title-btn"
          onClick={() => onNavigate?.('image')}
          aria-label="Return to image workspace"
          title="Return to image workspace"
        >
          <img
            src="/logo.png"
            alt="Concealed Logo"
            className="concealed-header-logo"
          />
          <span className="concealed-classic-title">CONCEALED</span>
        </button>
      </div>

      {/* Center: Workspace Navigation Tabs (IMAGE & VIDEO) */}
      <div className="topbar-center">
        <nav className="nav-segmented-group" role="tablist" aria-label="Workspace navigation">
          {/* IMAGE Tab */}
          <button
            type="button"
            role="tab"
            aria-selected={currentView === 'image'}
            className={`nav-segmented-tab ${currentView === 'image' ? 'active' : ''}`}
            onClick={() => onNavigate?.('image')}
          >
            IMAGE
          </button>

          {/* VIDEO Tab */}
          <button
            type="button"
            role="tab"
            aria-selected={currentView === 'video'}
            className={`nav-segmented-tab ${currentView === 'video' ? 'active' : ''}`}
            onClick={() => onNavigate?.('video')}
          >
            VIDEO
          </button>
        </nav>
      </div>

      <div className="topbar-right" />
    </header>
  );
};

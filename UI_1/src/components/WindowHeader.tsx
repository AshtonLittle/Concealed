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
        {/* Classic CONCEALED wordmark on the far left with stylized logo emblem */}
        <button
          type="button"
          className="concealed-title-btn"
          onClick={() => onNavigate?.('image')}
          aria-label="Return to image workspace"
          title="Return to image workspace"
        >
          <div className="topbar-logo-emblem" aria-hidden="true">
            <img src="/logo_white_cropped.png" alt="" className="topbar-logo-img" />
          </div>
          <span className="concealed-classic-title">CONCEALED</span>
        </button>

        {/* Excel-style Sheet Tabs Navigation */}
        <nav className="excel-tab-bar" role="tablist" aria-label="Worksheet pages">
          <span className="excel-nav-arrow" aria-hidden="true">›</span>

          {/* IMAGE Tab */}
          <button
            type="button"
            role="tab"
            aria-selected={currentView === 'image'}
            className={`excel-tab ${currentView === 'image' ? 'active' : ''}`}
            onClick={() => onNavigate?.('image')}
          >
            <span className="excel-tab-label">IMAGE</span>
          </button>

          <span className="excel-tab-separator" aria-hidden="true" />

          {/* VIDEO Tab */}
          <button
            type="button"
            role="tab"
            aria-selected={currentView === 'video'}
            className={`excel-tab ${currentView === 'video' ? 'active' : ''}`}
            onClick={() => onNavigate?.('video')}
          >
            <span className="excel-tab-label">VIDEO</span>
          </button>

          <span className="excel-tab-separator" aria-hidden="true" />

          {/* BENCHMARK Tab */}
          <button
            type="button"
            role="tab"
            aria-selected={currentView === 'benchmark'}
            className={`excel-tab ${currentView === 'benchmark' ? 'active' : ''}`}
            onClick={() => onNavigate?.('benchmark')}
          >
            <span className="excel-tab-label">BENCHMARK</span>
          </button>
        </nav>
      </div>

      {/* Right side is kept clean */}
      <div className="topbar-right" aria-hidden="true" />
    </header>
  );
};

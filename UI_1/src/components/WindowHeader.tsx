import React from 'react';

export type AppView = 'image' | 'video' | 'feature-obscuring' | 'probe' | 'benchmark';

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

      {/* Center: Workspace Navigation Tabs (Three Distinct Functional Groups) */}
      <div className="topbar-center">
        <div className="topbar-nav-groups" role="navigation" aria-label="Workspace navigation">
          {/* Tab Group 1: Global AI Concealing (IMAGE & VIDEO) */}
          <div className="nav-tab-group-container">
            <span className="nav-tab-group-label">AI Concealing</span>
            <div className="nav-segmented-group" role="tablist" aria-label="AI Concealing modes">
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
            </div>
          </div>

          {/* Visual Divider between Groups */}
          <div className="nav-groups-separator" aria-hidden="true" />

          {/* Tab Group 2: Targeted Feature Obscuring */}
          <div className="nav-tab-group-container">
            <span className="nav-tab-group-label">Targeted</span>
            <div className="nav-segmented-group nav-segmented-group-feature" role="tablist" aria-label="Feature Obscuring mode">
              {/* FEATURE OBSCURING Tab */}
              <button
                type="button"
                role="tab"
                aria-selected={currentView === 'feature-obscuring'}
                className={`nav-segmented-tab ${currentView === 'feature-obscuring' ? 'active' : ''}`}
                onClick={() => onNavigate?.('feature-obscuring')}
              >
                FEATURE OBSCURING
              </button>
            </div>
          </div>

          {/* Visual Divider between Groups */}
          <div className="nav-groups-separator" aria-hidden="true" />

          {/* Tab Group 3: Model Evasion Audit & Proof of Concept */}
          <div className="nav-tab-group-container">
            <span className="nav-tab-group-label">Evasion Proof</span>
            <div className="nav-segmented-group nav-segmented-group-probe" role="tablist" aria-label="Model Probe mode">
              {/* MODEL PROBE Tab */}
              <button
                type="button"
                role="tab"
                aria-selected={currentView === 'probe'}
                className={`nav-segmented-tab ${currentView === 'probe' ? 'active' : ''}`}
                onClick={() => onNavigate?.('probe')}
                title="Prompt Vision Transformers and VLMs to verify image understanding evasion"
              >
                MODEL PROBE
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="topbar-right" />
    </header>
  );
};

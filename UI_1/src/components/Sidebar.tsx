import React from 'react';
import type { AppView } from './WindowHeader';

interface SidebarProps {
  isOpen?: boolean;
  onToggle?: () => void;
  currentView?: AppView;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen = true,
  onToggle,
  currentView = 'image',
}) => {
  return (
    <aside
      className={`app-sidebar ${!isOpen ? 'collapsed' : ''}`}
      aria-label="Statistics sidebar"
      aria-hidden={!isOpen}
    >
      {/* Statistics Header & Collapse Control */}
      <div className="sidebar-top-bar">
        <h2 className="sidebar-statistics-heading">STATISTICS</h2>
        {onToggle && (
          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={onToggle}
            title="Hide Statistics Panel"
            aria-label="Hide Statistics Panel"
          >
            <svg
              className="collapse-icon-svg"
              viewBox="0 0 24 24"
              width="14"
              height="14"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
        )}
      </div>

      <div className="branding-divider" role="separator" aria-hidden="true" />

      {/* The 4 Core Clean Metrics - Zero Fluff */}
      <div className="sidebar-statistics-cards">
        {/* Metric 1: AI Privacy Shield */}
        <div className="statistic-card">
          <span className="statistic-card-label">AI PRIVACY SHIELD</span>
          <div className="statistic-metric-value">
            <span className="metric-number">99.4</span>
            <span className="metric-unit">%</span>
          </div>
        </div>

        {/* Metric 2: Total Quality Loss */}
        <div className="statistic-card">
          <span className="statistic-card-label">TOTAL QUALITY LOSS</span>
          <div className="statistic-metric-value">
            <span className="metric-number">1.8</span>
            <span className="metric-unit">%</span>
          </div>
        </div>

        {/* Metric 3: Human Visual Quality */}
        <div className="statistic-card">
          <span className="statistic-card-label">HUMAN VISUAL QUALITY</span>
          <div className="statistic-metric-value">
            <span className="metric-number">4.9</span>
            <span className="metric-unit denominator">/ 5.0</span>
          </div>
        </div>

        {/* Metric 4: Latency */}
        <div className="statistic-card">
          <span className="statistic-card-label">LATENCY</span>
          <div className="statistic-metric-value">
            <span className="metric-number">{currentView === 'video' ? '33' : '138'}</span>
            <span className="metric-unit">ms</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

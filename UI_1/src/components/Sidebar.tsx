import React from 'react';
import type { AppView } from './WindowHeader';

export interface ConcealStats {
  latencyMs?: number | null;
  qualityLossPct?: number | null;
  psnrDb?: number | null;
  ssim?: number | null;
}

interface SidebarProps {
  isOpen?: boolean;
  onToggle?: () => void;
  currentView?: AppView;
  stats?: ConcealStats | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen = true,
  onToggle,
  stats,
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

      {/* Live Measurement Cards: Total Quality Loss & Latency */}
      <div className="sidebar-statistics-cards">
        {/* Metric 1: Total Quality Loss (Normal vs Concealed) */}
        <div className="statistic-card">
          <span className="statistic-card-label">TOTAL QUALITY LOSS</span>
          <div className="statistic-metric-value">
            <span className="metric-number">
              {stats && stats.qualityLossPct !== undefined && stats.qualityLossPct !== null
                ? stats.qualityLossPct.toFixed(1)
                : '--'}
            </span>
            <span className="metric-unit">%</span>
          </div>
          <span className="statistic-card-sublabel">
            {stats && stats.qualityLossPct !== undefined && stats.qualityLossPct !== null
              ? `Degradation vs. clean image${stats.ssim ? ` (SSIM: ${stats.ssim.toFixed(3)})` : ''}`
              : 'Conceal image to measure quality loss'}
          </span>
        </div>

        {/* Metric 2: Live Conceal Latency */}
        <div className="statistic-card">
          <span className="statistic-card-label">LATENCY</span>
          <div className="statistic-metric-value">
            <span className="metric-number">
              {stats && stats.latencyMs !== undefined && stats.latencyMs !== null
                ? stats.latencyMs.toFixed(0)
                : '--'}
            </span>
            <span className="metric-unit">ms</span>
          </div>
          <span className="statistic-card-sublabel">
            {stats && stats.latencyMs !== undefined && stats.latencyMs !== null
              ? 'Synthesis wall-clock time'
              : 'Conceal image to measure latency'}
          </span>
        </div>
      </div>
    </aside>
  );
};

import React from 'react';
import { WindowHeader } from './WindowHeader';
import type { AppView } from './WindowHeader';

interface WindowFrameProps {
  children: React.ReactNode;
  currentView?: AppView;
  onNavigate?: (view: AppView) => void;
  isSidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  isSettingsOpen?: boolean;
  onToggleSettings?: () => void;
}

export const WindowFrame: React.FC<WindowFrameProps> = ({
  children,
  currentView = 'image',
  onNavigate,
  isSidebarOpen = true,
  onToggleSidebar,
  isSettingsOpen = true,
  onToggleSettings,
}) => {
  const isNoSidebarView = currentView === 'feature-obscuring' || currentView === 'probe';

  return (
    <div className="window-viewport">
      <div
        className="application-window"
        role="region"
        aria-label="Image Concealer Application"
      >
        {/* Top Window Bar */}
        <WindowHeader
          currentView={currentView}
          onNavigate={onNavigate}
        />

        {/* Window Body (Left Statistics + Middle Workspace + Right Settings) */}
        <div className="window-body">
          {/* Left Dock Button (Expand Statistics) */}
          {!isNoSidebarView && !isSidebarOpen && onToggleSidebar && (
            <button
              type="button"
              className="sidebar-dock-expand-btn"
              onClick={onToggleSidebar}
              title="Show Statistics Panel"
              aria-label="Show Statistics Panel"
            >
              <svg
                className="dock-icon-svg"
                viewBox="0 0 24 24"
                width="16"
                height="16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <polyline points="9 18 15 12 9 6" />
              </svg>
              <span className="dock-label">STATISTICS</span>
            </button>
          )}

          {children}

          {/* Right Dock Button (Expand Settings) */}
          {!isNoSidebarView && !isSettingsOpen && onToggleSettings && (
            <button
              type="button"
              className="settings-dock-expand-btn"
              onClick={onToggleSettings}
              title="Show Settings Panel"
              aria-label="Show Settings Panel"
            >
              <svg
                className="dock-icon-svg"
                viewBox="0 0 24 24"
                width="16"
                height="16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <polyline points="15 18 9 12 15 6" />
              </svg>
              <span className="dock-label">SETTINGS</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

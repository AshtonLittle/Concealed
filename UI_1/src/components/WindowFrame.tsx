import React from 'react';
import { WindowHeader } from './WindowHeader';
import type { AppView } from './WindowHeader';

interface WindowFrameProps {
  children: React.ReactNode;
  currentView?: AppView;
  onNavigate?: (view: AppView) => void;
}

export const WindowFrame: React.FC<WindowFrameProps> = ({
  children,
  currentView = 'image',
  onNavigate,
}) => {
  return (
    <div className="window-viewport">
      <div
        className="application-window"
        role="region"
        aria-label="Image Concealer Application"
      >
        {/* Top Window Bar with Excel-style Tabs */}
        <WindowHeader
          currentView={currentView}
          onNavigate={onNavigate}
        />

        {/* Window Body (Sidebar + Workspace / Video / Benchmark) */}
        <div className="window-body">
          {children}
        </div>
      </div>
    </div>
  );
};

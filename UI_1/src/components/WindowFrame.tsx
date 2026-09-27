import React from 'react';
import { WindowHeader } from './WindowHeader';

interface WindowFrameProps {
  children: React.ReactNode;
}

export const WindowFrame: React.FC<WindowFrameProps> = ({ children }) => {
  return (
    <div className="window-viewport">
      <div
        className="application-window"
        role="region"
        aria-label="Image Concealer Application"
      >
        {/* Top Window Bar */}
        <WindowHeader />

        {/* Window Body (Sidebar + Workspace) */}
        <div className="window-body">
          {children}
        </div>
      </div>
    </div>
  );
};

import React from 'react';
import { ProtectionStatus } from './ProtectionStatus';
import type { ConcealStats } from './Sidebar';

interface MainWorkspaceProps {
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const MainWorkspace: React.FC<MainWorkspaceProps> = ({ onStatsUpdate }) => {
  return (
    <main className="main-workspace" aria-label="Main protection workspace">
      <div className="workspace-content">
        <ProtectionStatus onStatsUpdate={onStatsUpdate} />
      </div>
    </main>
  );
};


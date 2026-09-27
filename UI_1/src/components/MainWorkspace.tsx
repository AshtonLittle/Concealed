import React from 'react';
import { ProtectionStatus } from './ProtectionStatus';
import type { ConcealStats } from './Sidebar';

interface MainWorkspaceProps {
  selectedModelEngine?: 'onnx' | 'pt';
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const MainWorkspace: React.FC<MainWorkspaceProps> = ({ selectedModelEngine = 'onnx', onStatsUpdate }) => {
  return (
    <main className="main-workspace" aria-label="Main protection workspace">
      <div className="workspace-content">
        <ProtectionStatus selectedModelEngine={selectedModelEngine} onStatsUpdate={onStatsUpdate} />
      </div>
    </main>
  );
};

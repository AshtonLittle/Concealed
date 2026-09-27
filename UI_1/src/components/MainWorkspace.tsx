import React from 'react';
import { ProtectionStatus } from './ProtectionStatus';
import type { ConcealStats } from './Sidebar';
import type { ProtectionMode, ImageOutputFormat } from './SettingsSidebar';

interface MainWorkspaceProps {
  selectedModelEngine?: 'onnx' | 'pt';
  budget?: number;
  mode?: ProtectionMode;
  outputFormat?: ImageOutputFormat;
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const MainWorkspace: React.FC<MainWorkspaceProps> = ({
  selectedModelEngine = 'onnx',
  budget = 8,
  mode = 'HYBRID',
  outputFormat = 'PNG',
  onStatsUpdate,
}) => {
  return (
    <main className="main-workspace" aria-label="Main protection workspace">
      <div className="workspace-content">
        <ProtectionStatus
          selectedModelEngine={selectedModelEngine}
          budget={budget}
          mode={mode}
          outputFormat={outputFormat}
          onStatsUpdate={onStatsUpdate}
        />
      </div>
    </main>
  );
};

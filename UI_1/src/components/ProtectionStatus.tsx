import React from 'react';
import { NoAISymbol } from './Icons';

export const ProtectionStatus: React.FC = () => {
  return (
    <div className="protection-status-center" aria-live="polite">
      {/* Visual Emblem: AI Knot with Diagonal Prohibition Slash */}
      <div className="emblem-container" aria-hidden="true">
        <NoAISymbol className="prohibition-emblem" />
      </div>

      {/* Primary Message */}
      <h2 className="protection-headline">
        NO AI. JUST RULES.
      </h2>

      {/* Subtitle / Status Line */}
      <div className="protection-status-line">
        <span className="status-indicator-dot" aria-hidden="true" />
        <span className="protection-submessage">
          IMAGE PROTECTION ACTIVE
        </span>
      </div>
    </div>
  );
};

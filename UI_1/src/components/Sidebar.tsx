import React from 'react';
import { Branding } from './Branding';
import { ImageSettings } from './ImageSettings';
import { PlaceholderOptions } from './PlaceholderOptions';

interface SidebarProps {
  resolution: string;
  onResolutionChange: (val: string) => void;
  format: string;
  onFormatChange: (val: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  resolution,
  onResolutionChange,
  format,
  onFormatChange,
}) => {
  return (
    <aside className="app-sidebar" aria-label="Control panel sidebar">
      <Branding />

      <ImageSettings
        resolution={resolution}
        onResolutionChange={onResolutionChange}
        format={format}
        onFormatChange={onFormatChange}
      />

      {/* Spacer to push lower tweaks section down naturally like in the reference */}
      <div className="sidebar-spacer" aria-hidden="true" />

      <PlaceholderOptions />
    </aside>
  );
};

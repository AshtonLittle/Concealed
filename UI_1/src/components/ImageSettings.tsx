import React from 'react';
import { SettingControl } from './SettingControl';
import { ShieldIcon, UserIcon } from './Icons';

interface ImageSettingsProps {
  resolution: string;
  onResolutionChange: (val: string) => void;
  format: string;
  onFormatChange: (val: string) => void;
}

const RESOLUTION_OPTIONS = ['720p', '1080p', '1440p', '2160p'];
const FORMAT_OPTIONS = ['PNG', 'JPEG', 'WEBP'];

export const ImageSettings: React.FC<ImageSettingsProps> = ({
  resolution,
  onResolutionChange,
  format,
  onFormatChange,
}) => {
  return (
    <section className="image-settings-section" aria-label="Image output settings">
      <SettingControl
        icon={<ShieldIcon size={20} />}
        label="Resolution"
        value={resolution}
        options={RESOLUTION_OPTIONS}
        onChange={onResolutionChange}
        hasChevron={false}
      />

      <SettingControl
        icon={<UserIcon size={20} />}
        label="Format"
        value={format}
        options={FORMAT_OPTIONS}
        onChange={onFormatChange}
        hasChevron={true}
      />
    </section>
  );
};

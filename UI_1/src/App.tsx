import { useState } from 'react';
import { WindowFrame } from './components/WindowFrame';
import { Sidebar } from './components/Sidebar';
import { MainWorkspace } from './components/MainWorkspace';
import './App.css';

export function App() {
  const [resolution, setResolution] = useState('1080p');
  const [format, setFormat] = useState('PNG');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  return (
    <WindowFrame>
      <Sidebar
        resolution={resolution}
        onResolutionChange={setResolution}
        format={format}
        onFormatChange={setFormat}
      />
      <MainWorkspace
        isSettingsOpen={isSettingsOpen}
        onToggleSettings={() => setIsSettingsOpen((prev) => !prev)}
        onCloseSettings={() => setIsSettingsOpen(false)}
      />
    </WindowFrame>
  );
}

export default App;

import { useState } from 'react';
import { WindowFrame } from './components/WindowFrame';
import { Sidebar } from './components/Sidebar';
import { MainWorkspace } from './components/MainWorkspace';
import { VideoWorkspace } from './components/VideoWorkspace';
import { BenchmarkView } from './components/BenchmarkView';
import type { AppView } from './components/WindowHeader';
import './App.css';

export function App() {
  const [currentView, setCurrentView] = useState<AppView>('image');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  return (
    <WindowFrame
      currentView={currentView}
      onNavigate={setCurrentView}
    >
      <Sidebar />
      {currentView === 'image' && (
        <MainWorkspace
          isSettingsOpen={isSettingsOpen}
          onToggleSettings={() => setIsSettingsOpen((prev) => !prev)}
          onCloseSettings={() => setIsSettingsOpen(false)}
        />
      )}
      {currentView === 'video' && (
        <VideoWorkspace
          isSettingsOpen={isSettingsOpen}
          onToggleSettings={() => setIsSettingsOpen((prev) => !prev)}
          onCloseSettings={() => setIsSettingsOpen(false)}
        />
      )}
      {currentView === 'benchmark' && (
        <BenchmarkView onBack={() => setCurrentView('image')} />
      )}
    </WindowFrame>
  );
}

export default App;

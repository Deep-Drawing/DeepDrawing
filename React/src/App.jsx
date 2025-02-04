import React from 'react';
import { EditorProvider } from './contexts/editorContext';
import CoordinateTracker from './components/TlDrawCanvasComponent/tldraw';
import './App.css';

function App() {
  return (
    <EditorProvider>
      <div className="App">
        <header className="App-header">
          {/* <h1>Music and Art Therapy App</h1> */}
        </header>
        <CoordinateTracker />
      </div>
    </EditorProvider>
  );
}

export default App;

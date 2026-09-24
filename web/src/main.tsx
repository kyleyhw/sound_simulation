import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';
import { useApp } from './state/store';

// Exposed for end-to-end tests and console exploration.
(window as unknown as { __app: typeof useApp }).__app = useApp;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

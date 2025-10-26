import React, { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WorkingThemeProvider } from './providers/WorkingThemeProvider';
import { Layout } from './components/Layout';
import Dashboard from './pages/Dashboard';
import Tasks from './pages/Tasks';
import Insights from './pages/Insights';
import Settings from './pages/Settings';
import Welcome from './pages/Welcome';
import EmotionAnalysis from './pages/EmotionAnalysis';
import './styles/theme.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  useEffect(() => {
    // Register service worker for PWA functionality
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js')
        .then((registration) => {
          console.log('SW registered: ', registration);
        })
        .catch((registrationError) => {
          console.log('SW registration failed: ', registrationError);
        });
    }
  }, []);

  return (
    <div className="App">
      <QueryClientProvider client={queryClient}>
        <WorkingThemeProvider>
          <Routes>
            <Route path="/welcome" element={<Welcome />} />
            <Route path="/*" element={
              <Layout>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/tasks" element={<Tasks />} />
                  <Route path="/emotion-analysis" element={<EmotionAnalysis />} />
                  <Route path="/insights" element={<Insights />} />
                  <Route path="/settings" element={<Settings />} />
                </Routes>
              </Layout>
            } />
          </Routes>
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'var(--color-surface)',
                color: 'var(--color-text)',
                border: '1px solid var(--color-primary)',
                borderRadius: 'var(--border-radius)',
                boxShadow: 'var(--shadow-md)',
              },
            }}
          />
        </WorkingThemeProvider>
      </QueryClientProvider>
    </div>
  );
}

export default App;
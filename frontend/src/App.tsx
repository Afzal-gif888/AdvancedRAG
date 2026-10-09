import { useEffect } from 'react';
import { Route, Routes } from 'react-router-dom';
import { useAppStore } from './store';
import LoginScreen from './components/LoginScreen';
import MainApp from './components/MainApp';
import { ForgotPasswordPage, ResetPasswordPage, SharedConversationPage } from './components/PublicPages';
import './index.css';

function useThemeSync() {
  const theme = useAppStore((s) => s.theme);
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: light)');
    const apply = () => {
      const resolved = theme === 'system' ? (media.matches ? 'light' : 'dark') : theme;
      document.documentElement.setAttribute('data-theme', resolved);
    };
    apply();
    media.addEventListener('change', apply);
    return () => media.removeEventListener('change', apply);
  }, [theme]);
}

function AuthGate() {
  const authStatus = useAppStore((s) => s.authStatus);
  const initAuth = useAppStore((s) => s.initAuth);

  useEffect(() => {
    // /auth/refresh is single-flight, so StrictMode's double effect is harmless.
    initAuth();
  }, [initAuth]);

  if (authStatus === 'loading') {
    return <div style={{ margin: 'auto', color: 'var(--text-secondary)' }}>Loading…</div>;
  }
  return authStatus === 'authenticated' ? <MainApp /> : <LoginScreen />;
}

export default function App() {
  useThemeSync();
  return (
    <Routes>
      <Route path="/share/:token" element={<SharedConversationPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route path="*" element={<AuthGate />} />
    </Routes>
  );
}

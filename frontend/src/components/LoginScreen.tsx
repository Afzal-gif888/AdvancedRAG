import { useState } from 'react';
import type { FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { api, errorMessage } from '../api';
import { useAppStore } from '../store';
import { inputStyle } from './helpers';
import { FormMessage } from './ui';

type Step = 'initial' | 'login' | 'register';

/**
 * Sign-up: name + email + password -> signed in.
 * Login:   email + password -> signed in.
 * Forgot password lives on its own page (/forgot-password).
 */
export default function LoginScreen() {
  const setSession = useAppStore((s) => s.setSession);
  const [step, setStep] = useState<Step>('initial');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const go = (next: Step) => {
    setStep(next);
    setError(null);
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (step === 'register') {
        setSession((await api.register(email, password, fullName.trim())).user);
      } else if (step === 'login') {
        setSession((await api.login(email, password)).user);
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const canSubmit = step === 'register' ? !!fullName.trim() && !!email && !!password : !!email && !!password;

  return (
    <div className="login-container">
      <div className="login-box">
        <h1 className="login-title">AdvancedRAG</h1>

        {step === 'initial' && (
          <>
            <button className="login-btn login-btn-primary" onClick={() => go('register')}>
              Sign up with email
            </button>
            <div className="login-divider">OR</div>
            <button className="login-btn login-btn-outline" onClick={() => go('login')}>
              Log in
            </button>
          </>
        )}

        {step !== 'initial' && (
          <form onSubmit={submit} noValidate>
            <p style={{ textAlign: 'center', marginBottom: '16px' }}>
              {step === 'register' && 'Create your account'}
              {step === 'login' && 'Enter your credentials to continue'}
            </p>

            {step === 'register' && (
              <input
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Full name"
                autoComplete="name"
                maxLength={100}
                autoFocus
                style={{ ...inputStyle, marginBottom: '12px' }}
              />
            )}
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email address"
              autoComplete="email"
              autoFocus={step !== 'register'}
              style={{ ...inputStyle, marginBottom: '12px' }}
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              autoComplete={step === 'register' ? 'new-password' : 'current-password'}
              maxLength={128}
              style={inputStyle}
            />
            {step === 'register' && (
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '8px' }}>
                At least 8 characters, including a letter and a number.
              </div>
            )}

            <FormMessage error={error} />

            <button
              type="submit"
              className="login-btn login-btn-primary"
              disabled={busy || !canSubmit}
              style={{ marginTop: '16px', opacity: busy || !canSubmit ? 0.5 : 1 }}
            >
              {busy ? 'Please wait…' : step === 'register' ? 'Create account' : 'Log in'}
            </button>

            {step === 'login' && (
              <Link to="/forgot-password" state={{ email }} className="login-btn login-btn-outline" style={{ border: 'none', marginTop: '8px', textAlign: 'center' }}>
                Forgot password?
              </Link>
            )}
            {step === 'register' && (
              <button type="button" className="login-btn login-btn-outline" onClick={() => go('login')} style={{ border: 'none', marginTop: '8px' }}>
                Already have an account? Log in
              </button>
            )}
            <button
              type="button"
              className="login-btn login-btn-outline"
              onClick={() => go('initial')}
              style={{ border: 'none', marginTop: '8px' }}
            >
              Back
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

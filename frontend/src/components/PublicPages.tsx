import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { Link, useLocation, useParams, useSearchParams } from 'react-router-dom';
import { api, ApiError, errorMessage } from '../api';
import type { SharedConversation } from '../types';
import { inputStyle } from './helpers';
import { FormMessage } from './ui';

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="login-container">
      <div className="login-box">
        <h1 className="login-title">AdvancedRAG</h1>
        {children}
      </div>
    </div>
  );
}

const RESET_TOKEN = /^[A-Za-z0-9_-]{32,128}$/;  // shape of the backend's opaque reset tokens
const INVALID_LINK = 'This reset link is invalid or has expired.';

function passwordProblem(password: string, confirm: string): string | null {
  if (password.length < 8) return 'Password must be at least 8 characters.';
  if (password.length > 128) return 'Password must be at most 128 characters.';
  if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) return 'Password must contain at least one letter and one number.';
  if (password.trim() !== password) return 'Password must not start or end with whitespace.';
  if (password !== confirm) return 'Passwords do not match.';
  return null;
}

/** Only messages the API defines for users are shown; anything else gets a generic one. */
function safeError(err: unknown): string {
  if (err instanceof ApiError && err.status < 500) return errorMessage(err);
  if (err instanceof ApiError) return 'Something went wrong. Please try again in a moment.';
  return errorMessage(err);  // network error message
}

const linkStyle = { marginTop: 16, textAlign: 'center' as const };

export function ForgotPasswordPage() {
  const location = useLocation();
  const [email, setEmail] = useState<string>((location.state as { email?: string } | null)?.email ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setSent((await api.forgotPassword(email.trim())).message);
    } catch (err) {
      setError(safeError(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Centered>
      {sent ? (
        <>
          <FormMessage success={sent} />
          <p style={{ textAlign: 'center', marginTop: 12, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Check your inbox and spam folder. The link can be used once.
          </p>
          <Link to="/" className="login-btn login-btn-primary" style={linkStyle}>Back to sign in</Link>
        </>
      ) : (
        <form onSubmit={submit} noValidate>
          <p style={{ textAlign: 'center', marginBottom: 16 }}>Enter your email and we'll send you a reset link</p>
          <input id="reset-email" name="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email address" autoComplete="email" style={inputStyle} autoFocus />
          <FormMessage error={error} />
          <button type="submit" className="login-btn login-btn-primary" disabled={busy || !email.trim()} style={{ marginTop: 16, opacity: busy || !email.trim() ? 0.5 : 1 }}>
            {busy ? 'Sending…' : 'Send reset link'}
          </button>
          <Link to="/" className="login-btn login-btn-outline" style={{ ...linkStyle, marginTop: 8, border: 'none' }}>Back to sign in</Link>
        </form>
      )}
    </Centered>
  );
}

export function ResetPasswordPage() {
  const [params] = useSearchParams();
  const token = params.get('token') ?? '';
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [invalidLink, setInvalidLink] = useState(!RESET_TOKEN.test(token));
  const [done, setDone] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const problem = passwordProblem(password, confirm);
    if (problem) {
      setError(problem);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.resetPassword(token, password);
      setDone(true);
    } catch (err) {
      if (err instanceof ApiError && err.code === 'INVALID_TOKEN') setInvalidLink(true);
      else setError(safeError(err));
    } finally {
      setBusy(false);
    }
  };

  if (invalidLink) {
    return (
      <Centered>
        <FormMessage error={INVALID_LINK} />
        <p style={{ textAlign: 'center', marginTop: 12, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Reset links expire and can be used once. Request a new one to continue.
        </p>
        <Link to="/forgot-password" className="login-btn login-btn-primary" style={linkStyle}>Request a new link</Link>
        <Link to="/" className="login-btn login-btn-outline" style={{ ...linkStyle, marginTop: 8, border: 'none' }}>Back to sign in</Link>
      </Centered>
    );
  }

  return (
    <Centered>
      {done ? (
        <>
          <FormMessage success="Your password has been reset. Sign in with your new password." />
          <Link to="/" className="login-btn login-btn-primary" style={linkStyle}>Go to sign in</Link>
        </>
      ) : (
        <form onSubmit={submit} noValidate>
          <p style={{ textAlign: 'center', marginBottom: 16 }}>Choose a new password</p>
          <input id="new-password" name="new-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="New password" autoComplete="new-password" maxLength={128} style={{ ...inputStyle, marginBottom: 12 }} autoFocus />
          <input id="confirm-password" name="confirm-password" type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="Confirm new password" autoComplete="new-password" maxLength={128} style={inputStyle} />
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 8 }}>
            At least 8 characters, including a letter and a number.
          </div>
          <FormMessage error={error} />
          <button type="submit" className="login-btn login-btn-primary" disabled={busy || !password || !confirm} style={{ marginTop: 16, opacity: busy || !password || !confirm ? 0.5 : 1 }}>
            {busy ? 'Saving…' : 'Reset password'}
          </button>
          <Link to="/" className="login-btn login-btn-outline" style={{ ...linkStyle, marginTop: 8, border: 'none' }}>Back to sign in</Link>
        </form>
      )}
    </Centered>
  );
}

export function SharedConversationPage() {
  const { token = '' } = useParams();
  const [data, setData] = useState<SharedConversation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getSharedConversation(token).then(setData).catch((err) => setError(errorMessage(err)));
  }, [token]);

  return (
    <div className="app-container">
      <div className="main-chat">
        <div className="chat-header">
          <div className="header-left">
            <h2 style={{ fontSize: '1rem' }}>{data?.title ?? 'Shared conversation'}</h2>
          </div>
          <div className="header-actions">
            <Link to="/" className="login-btn login-btn-outline" style={{ padding: '6px 12px', width: 'auto' }}>Open AdvancedRAG</Link>
          </div>
        </div>
        <div className="messages-container">
          {error && <div style={{ margin: 'auto' }}><FormMessage error={error} /></div>}
          {!data && !error && <p style={{ margin: 'auto', color: 'var(--text-secondary)' }}>Loading…</p>}
          {data && (
            <>
              <p style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                Shared on {new Date(data.shared_at).toLocaleDateString()} · read-only
              </p>
              {data.messages.map((m, i) => (
                <div key={i} className="message-wrapper">
                  <div className={`message-avatar ${m.role}`}>{m.role === 'user' ? 'U' : 'AG'}</div>
                  <div className="message-content" style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

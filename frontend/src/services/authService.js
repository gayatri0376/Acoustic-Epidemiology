// Authentication logic: demo mode OR the real Django backend (never mixed).
import { apiRequest, ApiError, setAccessToken } from './api';
import { DEMO_MODE, ENDPOINTS, AUTH_CONFIG, ROLES } from '../config/config';
import { DEMO_USERS, DEMO_PASSWORD } from './demoData';

const DEMO_KEY = 'ae_demo_user'; // demo only: synthetic user, never a real credential

// >>> EDIT HERE after verifying your serializers <<<
// Turns whatever the backend returns into { id, name, email, role }.
// The field names (user, role, full_name ...) are ASSUMPTIONS until you confirm them.
// The role must come from the server's response, never from the login form.
export function normalizeUser(data) {
  const u = data?.user ?? data;
  let role = String(u?.role ?? u?.profile?.role ?? '').toLowerCase();
  if (role === 'clinician' || role === 'researcher') role = 'doctor';
  if (role === 'administrator') role = 'admin';
  if (role === 'uploader') role = 'patient';

  if (!ROLES.includes(role)) {
    throw new ApiError(
      `The server response role "${role}" is not recognised (patient, doctor or admin).`,
      { kind: 'config' },
    );
  }
  return { id: u.id, name: u.full_name ?? u.name ?? u.username ?? u.email, email: u.email, role };
}

function requireEndpoint(key, envName) {
  if (!ENDPOINTS[key]) {
    throw new ApiError(
      `Login is not configured: set ${envName} in .env to your verified backend path, or enable VITE_DEMO_MODE to preview the UI.`,
      { kind: 'config' },
    );
  }
}

async function fetchCurrentUser() {
  return normalizeUser(await apiRequest(ENDPOINTS.me));
}

export async function login(identifier, password) {
  if (DEMO_MODE) {
    await new Promise((r) => setTimeout(r, 500)); // mimic network delay
    const user = DEMO_USERS[identifier.trim().toLowerCase()];
    if (!user || password !== DEMO_PASSWORD) {
      throw new ApiError('Invalid email or password.', { status: 401 });
    }
    sessionStorage.setItem(DEMO_KEY, JSON.stringify(user));
    return user;
  }

  requireEndpoint('login', 'VITE_AUTH_LOGIN_PATH');
  const data = await apiRequest(ENDPOINTS.login, {
    method: 'POST',
    auth: false,
    body: { [AUTH_CONFIG.identifierField]: identifier, password },
  });

  if (AUTH_CONFIG.mode === 'token') {
    if (!AUTH_CONFIG.tokenKey || !AUTH_CONFIG.tokenScheme) {
      throw new ApiError('Token mode needs VITE_AUTH_TOKEN_KEY and VITE_AUTH_TOKEN_SCHEME.', { kind: 'config' });
    }
    if (!data?.[AUTH_CONFIG.tokenKey]) {
      throw new ApiError('Login response did not contain the expected token field.', { kind: 'config' });
    }
    setAccessToken(data[AUTH_CONFIG.tokenKey]);
  }
  // Prefer asking the backend "who am I?" so the role is read from trusted server data.
  return ENDPOINTS.me ? fetchCurrentUser() : normalizeUser(data);
}

export async function logout() {
  if (DEMO_MODE) { sessionStorage.removeItem(DEMO_KEY); return; }
  setAccessToken(null);
  if (!ENDPOINTS.logout) return;
  try { await apiRequest(ENDPOINTS.logout, { method: 'POST' }); } catch { /* still sign out locally */ }
}

// Called once on page load to restore a session.
export async function restoreSession() {
  if (DEMO_MODE) {
    try { return JSON.parse(sessionStorage.getItem(DEMO_KEY)); } catch { return null; }
  }
  if (!ENDPOINTS.me) return null; // cannot restore without a "current user" endpoint
  try { return await fetchCurrentUser(); } catch { return null; }
}

export async function checkHealth() {
  return apiRequest(ENDPOINTS.health, { auth: false });
}

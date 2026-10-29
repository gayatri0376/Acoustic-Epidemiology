// Central API client. Every request to Django goes through apiRequest().
import { API_BASE_URL, AUTH_CONFIG } from '../config/config';

// A friendly error type so the UI can react to the kind of failure.
// kind: 'config' | 'network' | 'http'
export class ApiError extends Error {
  constructor(message, { kind = 'http', status = null, data = null } = {}) {
    super(message);
    this.kind = kind;
    this.status = status;
    this.data = data;
  }
}

// Token mode only: the token lives in memory (not localStorage), so it is lost on refresh.
let accessToken = null;
export const setAccessToken = (t) => { accessToken = t; };

// Called when the server answers 401 for a signed-in user (expired session).
let onUnauthorized = null;
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn; };

function getCookie(name) {
  const match = document.cookie.split('; ').find((c) => c.startsWith(`${name}=`));
  return match ? decodeURIComponent(match.split('=')[1]) : null;
}

// Pull a readable message out of a Django REST Framework error body.
function messageFromData(data, fallback) {
  if (!data || typeof data !== 'object') return fallback;
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.non_field_errors)) return data.non_field_errors[0];
  return fallback;
}

export async function apiRequest(path, { method = 'GET', body, auth = true } = {}) {
  if (!API_BASE_URL) {
    throw new ApiError('VITE_API_BASE_URL is not set. Check your .env file.', { kind: 'config' });
  }
  const headers = { Accept: 'application/json' };
  if (body) headers['Content-Type'] = 'application/json';

  if (AUTH_CONFIG.mode === 'token' && auth && accessToken && AUTH_CONFIG.tokenScheme) {
    headers.Authorization = `${AUTH_CONFIG.tokenScheme} ${accessToken}`;
  }
  // Django session auth requires the CSRF token on unsafe methods.
  if (AUTH_CONFIG.mode === 'cookie' && method !== 'GET') {
    const csrf = getCookie('csrftoken');
    if (csrf) headers['X-CSRFToken'] = csrf;
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      credentials: AUTH_CONFIG.mode === 'cookie' ? 'include' : 'omit',
    });
  } catch {
    throw new ApiError(
      'Cannot reach the server. Check that the backend is running and that CORS allows this origin.',
      { kind: 'network' },
    );
  }

  let data = null;
  try { data = await response.json(); } catch { /* empty or non-JSON body */ }

  if (!response.ok) {
    if (response.status === 401 && auth && onUnauthorized) onUnauthorized();
    throw new ApiError(messageFromData(data, `Request failed (${response.status}).`), {
      status: response.status,
      data,
    });
  }
  return data;
}

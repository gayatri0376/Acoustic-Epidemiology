// ONE place for all environment-driven settings. Components never read import.meta.env directly.
const env = import.meta.env;

export const API_BASE_URL = (env.VITE_API_BASE_URL || '').replace(/\/$/, '');
export const DEMO_MODE = env.VITE_DEMO_MODE === 'true';

// Endpoint paths. Auth paths are intentionally EMPTY by default: they must be copied
// from your Django urls.py after you verify them (see README, "Backend checklist").
export const ENDPOINTS = {
  health: '/api/v1/health/', // path supplied by the project owner; not verified by this build
  login: env.VITE_AUTH_LOGIN_PATH || null,
  logout: env.VITE_AUTH_LOGOUT_PATH || null,
  me: env.VITE_AUTH_ME_PATH || null,
};

export const AUTH_CONFIG = {
  mode: env.VITE_AUTH_MODE === 'token' ? 'token' : 'cookie',
  identifierField: env.VITE_AUTH_IDENTIFIER_FIELD || 'email',
  tokenKey: env.VITE_AUTH_TOKEN_KEY || null,
  tokenScheme: env.VITE_AUTH_TOKEN_SCHEME || null,
};

export const ROLES = ['patient', 'doctor', 'admin'];
export const ROLE_HOME = {
  patient: '/patient/dashboard',
  doctor: '/doctor/dashboard',
  admin: '/admin/dashboard',
};

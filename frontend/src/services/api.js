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
<<<<<<< HEAD

/**
 * Retrieves the documented API endpoint catalog via GET /api/v1/
 */
export async function getApiIndex() {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Authenticates user credentials returning token and profile
 */
export async function loginUser(username, password) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/auth/login/`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Registers new uploader user
 */
export async function registerUser(registrationData) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/auth/register/`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(registrationData),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Uploads audio recording via multipart/form-data with token authorization
 */
export async function uploadAudioRecording(formData, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/recordings/`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: getAuthHeaders(token, true),
      body: formData,
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Fetches recordings permitted to the authenticated user
 */
export async function fetchRecordings(token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/recordings/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: getAuthHeaders(token),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Requests ML inference for a recording
 * Handles HTTP 503 (Model Unavailable), HTTP 400 (Invalid Audio), and HTTP 500 cleanly.
 */
export async function requestPrediction(recordingId, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/recordings/${recordingId}/predict/`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: getAuthHeaders(token),
    });
    const data = await response.json();
    return {
      success: response.ok,
      data,
      statusCode: response.status,
      isModelUnavailable: response.status === 503,
      isInvalidAudio: response.status === 400,
      errorMessage: response.ok ? null : (data.error || data.detail || data.message || 'Inference request failed'),
    };
  } catch (err) {
    return {
      success: false,
      error: err.message,
      isNetworkError: true,
      errorMessage: `Network error connecting to inference endpoint: ${err.message}`,
    };
  }
}

/**
 * Retrieves a single prediction by ID
 */
export async function fetchPrediction(predictionId, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/predictions/${predictionId}/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: getAuthHeaders(token),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Retrieves all prediction records for a specific recording
 */
export async function fetchRecordingPredictions(recordingId, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/recordings/${recordingId}/predictions/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: getAuthHeaders(token),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Requests Grad-CAM explanation for a prediction
 * Handles HTTP 503 (Unsupported / Weights missing) cleanly without crashing.
 */
export async function requestExplanation(predictionId, token, method = 'GRAD_CAM') {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/predictions/${predictionId}/explanations/`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: getAuthHeaders(token),
      body: JSON.stringify({ method }),
    });
    const data = await response.json();
    return {
      success: response.ok,
      data,
      statusCode: response.status,
      isUnsupported: response.status === 503,
      errorMessage: response.ok ? null : (data.error || data.message || data.detail || 'Explanation request failed'),
    };
  } catch (err) {
    return {
      success: false,
      error: err.message,
      isNetworkError: true,
      errorMessage: `Network error requesting explanation: ${err.message}`,
    };
  }
}

/**
 * Retrieves all generated explanations for a prediction
 */
export async function fetchPredictionExplanations(predictionId, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/predictions/${predictionId}/explanations/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: getAuthHeaders(token),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Retrieves explanation details and artifact URL by ID
 */
export async function fetchExplanation(explanationId, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/explanations/${explanationId}/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: getAuthHeaders(token),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Fetches the currently active model from the registry
 */
export async function fetchActiveModel() {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/models/active/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    const data = await response.json();
    return {
      success: response.ok,
      data,
      statusCode: response.status,
      isModelDeployed: response.ok,
    };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Fetches all registered models from the registry
 */
export async function fetchModels() {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/models/`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Submits clinician human review
 */
export async function submitAssessment(predictionId, assessmentData, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/predictions/${predictionId}/assessments/`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: getAuthHeaders(token),
      body: JSON.stringify(assessmentData),
    });
    const data = await response.json();
    return { success: response.ok, data, statusCode: response.status };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

export { API_BASE_URL };
=======
>>>>>>> c882e3b4a4911c3b7380c1be5b04e403e0abfafe

/**
 * API Service for Acoustic Epidemiology
 * Communicates with the Django REST Framework backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

function getAuthHeaders(token, isMultipart = false) {
  const headers = {
    'Accept': 'application/json',
  };
  if (!isMultipart) {
    headers['Content-Type'] = 'application/json';
  }
  if (token) {
    headers['Authorization'] = `Token ${token}`;
  }
  return headers;
}

/**
 * Checks backend liveness status via GET /api/v1/health/
 */
export async function checkBackendHealth() {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/health/`;
  const startTime = performance.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const latency = Math.round(performance.now() - startTime);

    if (!response.ok) {
      return {
        success: false,
        error: `Server responded with HTTP ${response.status}: ${response.statusText}`,
        statusCode: response.status,
        latency,
        url,
      };
    }

    const data = await response.json();
    return {
      success: true,
      data,
      statusCode: response.status,
      latency,
      url,
    };
  } catch (err) {
    const latency = Math.round(performance.now() - startTime);
    let errorMessage = err.message;
    if (err.name === 'AbortError') {
      errorMessage = 'Request timed out after 8 seconds. Please check if the Django server is running.';
    } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      errorMessage = `Cannot connect to Django server at ${url}. Ensure the server is running on http://127.0.0.1:8000.`;
    }

    return {
      success: false,
      error: errorMessage,
      latency,
      url,
    };
  }
}

/**
 * Checks backend readiness status via GET /api/v1/ready/
 */
export async function checkBackendReadiness() {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/ready/`;
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
 */
export async function requestPrediction(recordingId, token) {
  const url = `${API_BASE_URL.replace(/\/+$/, '')}/api/v1/recordings/${recordingId}/predict/`;
  try {
    const response = await fetch(url, {
      method: 'POST',
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

import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import * as authService from '../services/authService';
import { setUnauthorizedHandler, ApiError } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true); // true while restoring the session

  useEffect(() => {
    let cancelled = false;
    authService.restoreSession().then((u) => {
      if (!cancelled) { setUser(u); setLoading(false); }
    });
    // If the server says "401" later, drop the session so the user is sent to /login.
    setUnauthorizedHandler(() => setUser(null));
    return () => { cancelled = true; };
  }, []);

  // expectedRole is what the user clicked on the form. It is only a CHECK against the
  // role the server reports; it never grants anything.
  const login = useCallback(async (identifier, password, expectedRole) => {
    const verified = await authService.login(identifier, password);
    if (verified.role !== expectedRole) {
      await authService.logout();
      throw new ApiError(
        `This account is registered as ${verified.role}, not ${expectedRole}. Choose the correct role and try again.`,
        { kind: 'role' },
      );
    }
    setUser(verified);
    return verified;
  }, []);

  const logout = useCallback(async () => {
    await authService.logout();
    setUser(null);
  }, []);

  return <AuthContext.Provider value={{ user, loading, login, logout }}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);

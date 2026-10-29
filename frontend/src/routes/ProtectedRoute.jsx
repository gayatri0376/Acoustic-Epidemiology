import { Navigate, useLocation } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

// Usability layer only. Django must still enforce permissions on every API call.
export default function ProtectedRoute({ allowedRoles, children }) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="full-center" role="status">
        <Loader2 className="spin" size={28} aria-hidden="true" />
        <p>Restoring your session…</p>
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace state={{ from: location }} />;
  if (!allowedRoles.includes(user.role)) return <Navigate to="/unauthorized" replace />;
  return children;
}

import { Link } from 'react-router-dom';
import { ShieldAlert } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { ROLE_HOME } from '../config/config';

export default function Unauthorized() {
  const { user } = useAuth();
  return (
    <div className="full-center">
      <ShieldAlert size={40} color="#b42318" aria-hidden="true" />
      <h1>Access denied</h1>
      <p>Your account does not have permission to view that page.</p>
      <Link className="btn btn-primary" to={user ? ROLE_HOME[user.role] : '/login'}>
        {user ? 'Go to my dashboard' : 'Go to sign in'}
      </Link>
    </div>
  );
}

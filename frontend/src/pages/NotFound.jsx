import { Link } from 'react-router-dom';
import { SearchX } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="full-center">
      <SearchX size={40} color="#117f86" aria-hidden="true" />
      <h1>Page not found</h1>
      <p>The page you are looking for does not exist.</p>
      <Link className="btn btn-primary" to="/">Back to home</Link>
    </div>
  );
}

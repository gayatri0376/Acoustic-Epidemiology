import { DatabaseZap } from 'lucide-react';

// Shown when there is no data. In real mode this means "backend not connected yet".
export default function EmptyState({ title = 'No data available', children }) {
  return (
    <div className="empty">
      <DatabaseZap size={26} aria-hidden="true" />
      <strong>{title}</strong>
      <span>{children}</span>
    </div>
  );
}

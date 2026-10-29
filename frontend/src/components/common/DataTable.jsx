// Minimal table. columns: [{ key, label, render? }]
export default function DataTable({ columns, rows, caption }) {
  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>{columns.map((c) => <th key={c.key} scope="col">{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              {columns.map((c) => <td key={c.key}>{c.render ? c.render(row) : row[c.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Colour-coded status pill.
export function StatusBadge({ status }) {
  const cls = status === 'Completed' ? 'ok' : status === 'Pending upload' ? 'warn' : '';
  return <span className={`badge ${cls}`}>{status}</span>;
}

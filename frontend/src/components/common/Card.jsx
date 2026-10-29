// A titled white card with an icon. "id" lets sidebar links scroll to it.
export default function Card({ id, icon: Icon, title, children }) {
  return (
    <section className="card section" id={id} aria-labelledby={id ? `${id}-title` : undefined}>
      <div className="card-head">
        {Icon && <Icon size={20} aria-hidden="true" />}
        <h2 id={id ? `${id}-title` : undefined}>{title}</h2>
      </div>
      {children}
    </section>
  );
}

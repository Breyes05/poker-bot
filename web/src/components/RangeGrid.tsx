interface RangeGridProps {
  classes: string[]; // 169 names in chart order (row * 13 + col)
  values: number[]; // probability per class
  title: string;
  highlight?: string;
}

export function RangeGrid({ classes, values, title, highlight }: RangeGridProps) {
  return (
    <figure className="range">
      <figcaption>{title}</figcaption>
      <div className="range-grid" role="table" aria-label={title}>
        {classes.map((name, i) => {
          const p = values[i] ?? 0;
          const mixed = p > 0.005 && p < 0.995;
          return (
            <div
              key={name}
              role="cell"
              className={`range-cell ${name === highlight ? "range-highlight" : ""} ${p > 0.55 ? "range-dark" : ""}`}
              style={{ "--p": `${Math.round(p * 100)}%` } as React.CSSProperties}
              title={`${name}: ${(p * 100).toFixed(1)}%`}
            >
              <span>{name}</span>
              {mixed && <small>{Math.round(p * 100)}</small>}
            </div>
          );
        })}
      </div>
    </figure>
  );
}

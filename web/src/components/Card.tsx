const SUIT_SYMBOLS: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };

interface CardProps {
  card?: string | null; // e.g. "As", "Td"; missing = face down
  size?: "md" | "lg";
}

export function Card({ card, size = "lg" }: CardProps) {
  if (!card) {
    return <div className={`card card-${size} card-back`} aria-label="face-down card" />;
  }
  const rank = card[0] === "T" ? "10" : card[0];
  const suit = card[1];
  const red = suit === "h" || suit === "d";
  return (
    <div className={`card card-${size} ${red ? "card-red" : ""}`} aria-label={card}>
      <span className="card-rank">{rank}</span>
      <span className="card-suit">{SUIT_SYMBOLS[suit]}</span>
    </div>
  );
}

export function CardRow({ cards, count }: { cards: string[] | null; count: number }) {
  const shown = cards ?? Array<string | null>(count).fill(null);
  return (
    <div className="card-row">
      {shown.map((c, i) => (
        <Card key={i} card={c} />
      ))}
    </div>
  );
}

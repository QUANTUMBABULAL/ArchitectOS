/**
 * Segmented progress indicator.
 *
 * Rendered as discrete blocks rather than a smooth bar: the engine drives
 * a real browser and cannot report true token-level progress, so showing
 * coarse stages is honest where a precise percentage would not be.
 */

interface Props {
  value: number;
  segments?: number;
  tone?: "accent" | "ok" | "warn" | "bad";
  className?: string;
}

const TONE: Record<string, string> = {
  accent: "bg-accent",
  ok: "bg-ok",
  warn: "bg-warn",
  bad: "bg-bad",
};

export default function ProgressBar({
  value,
  segments = 10,
  tone = "accent",
  className = "",
}: Props) {
  const filled = Math.round(Math.max(0, Math.min(1, value)) * segments);

  return (
    <div className={`flex gap-[3px] ${className}`}>
      {Array.from({ length: segments }, (_, index) => (
        <div
          key={index}
          className={`h-1.5 flex-1 rounded-full transition-all duration-500 ${
            index < filled ? TONE[tone] : "bg-white/[.07]"
          }`}
        />
      ))}
    </div>
  );
}

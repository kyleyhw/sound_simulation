import { Brush, Circle, Eraser, Mic, Minus, MousePointer2, Square, Volume2, Waves } from 'lucide-react';
import type { ComponentType } from 'react';
import { type Tool, useApp } from '../state/store';

const TOOLS: { id: Tool; label: string; key: string; Icon: ComponentType<{ size?: number }> }[] = [
  { id: 'select', label: 'Select / move', key: 'V', Icon: MousePointer2 },
  { id: 'brush', label: 'Brush (paint walls)', key: 'B', Icon: Brush },
  { id: 'eraser', label: 'Eraser', key: 'E', Icon: Eraser },
  { id: 'line', label: 'Line wall', key: 'L', Icon: Minus },
  { id: 'rect', label: 'Rectangle (Shift = filled)', key: 'R', Icon: Square },
  { id: 'ellipse', label: 'Ellipse (Shift = filled)', key: 'O', Icon: Circle },
  { id: 'speed', label: 'Sound-speed brush (lenses, gradients)', key: 'C', Icon: Waves },
  { id: 'driver', label: 'Place source', key: 'S', Icon: Volume2 },
  { id: 'probe', label: 'Place microphone', key: 'M', Icon: Mic },
];

export const TOOL_KEYS: Record<string, Tool> = Object.fromEntries(TOOLS.map((t) => [t.key.toLowerCase(), t.id]));

export function ToolRail() {
  const tool = useApp((s) => s.tool);
  const setTool = useApp((s) => s.setTool);
  return (
    <nav className="rail" aria-label="Tools">
      {TOOLS.map(({ id, label, key, Icon }, i) => (
        <span key={id} style={{ display: 'contents' }}>
          {(i === 1 || i === 7) && <span className="sep" aria-hidden />}
          <button
            className="tool"
            aria-pressed={tool === id}
            aria-label={label}
            title={`${label} (${key})`}
            data-tool={id}
            onClick={() => setTool(id)}
          >
            <Icon size={19} />
          </button>
        </span>
      ))}
    </nav>
  );
}

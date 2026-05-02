"""
TikZ generators for hardware illustrations:
- Register bit field layouts
- Memory maps
- Timing diagrams
"""

import re


def _bit_field_row_tikz(bits: list[dict], width: int = 32) -> str:
    r"""Generate a TikZ register bit field diagram.

    Input: list of dicts with 'bits' (e.g., '31' or '23-16') and 'name'.
    Output: complete TikZ picture code (inside \begin{tikzpicture}...\end{tikzpicture}).
    """
    # Parse bit ranges
    fields = []
    for b in bits:
        br = b.get('bits', '').strip()
        name = b.get('name', '?')
        if '-' in br:
            hi, lo = br.split('-')
            hi_i, lo_i = int(hi), int(lo)
            fields.append((hi_i, lo_i, name))
        else:
            n = int(br)
            fields.append((n, n, name))

    cell_w = 0.5  # cm per bit
    height = 0.8  # cm
    total_w = width * cell_w

    lines = [r"\begin{center}",
             r"\begin{tikzpicture}[x=" + str(cell_w) + r"cm, y=" + str(height) + r"cm, every node/.style={font=\footnotesize}]",
             # Draw frame
             f"\\draw[thick] (0,0) rectangle ({width},1);"]

    # Field separators + labels
    for hi, lo, name in fields:
        left = width - hi - 1
        right = width - lo
        w = right - left
        # Draw field background
        if w > 1:
            lines.append(f"\\draw[thick, fill=blue!15] ({left},0) rectangle ({right},1);")
        else:
            lines.append(f"\\draw[thick, fill=blue!25] ({left},0) rectangle ({right},1);")
        # Label in center
        cx = (left + right) / 2
        lines.append(f"\\node at ({cx},0.5) {{{name}}};")

    # Bit numbers above
    for i in range(width):
        lines.append(f"\\node[above] at ({width - i - 0.5}, 1) {{\\tiny {i}}};")

    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{center}")
    return "\n".join(lines)


def parse_markdown_register_table(table_md: str) -> list[dict] | None:
    r"""Parse a markdown table describing register bits.

    Expects format like:
    | Bits  | Name | Description |
    |-------|------|-------------|
    | 31    | EN   | Enable      |
    | 30-24 | RSVD | Reserved    |

    Returns list of {bits, name, description} or None.
    """
    lines = [l.strip() for l in table_md.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return None

    # Find header columns
    header = [c.strip().lower() for c in lines[0].strip("|").split("|")]
    bits_col = None
    name_col = None
    for i, h in enumerate(header):
        if 'bit' in h:
            bits_col = i
        elif 'name' in h or 'champ' in h or 'field' in h:
            name_col = i

    if bits_col is None or name_col is None:
        return None

    # Skip separator line
    fields = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) > max(bits_col, name_col):
            fields.append({
                "bits": cells[bits_col],
                "name": cells[name_col],
            })

    return fields if fields else None


def detect_and_convert_register_tables(markdown_text: str) -> str:
    r"""Scan markdown for register bit-field tables and add TikZ diagrams above them.

    Detection: tables where first column header is 'Bits' or 'Bit'.
    """
    # Find markdown tables with 'Bits' or 'Bit' as first column header
    pattern = re.compile(
        r'(\|\s*[Bb]it[s]?\s*\|[^\n]*\n\|[\s\-|:]+\|\n(?:\|[^\n]*\|\n?)+)',
        re.MULTILINE
    )

    def replace(match):
        table = match.group(1)
        fields = parse_markdown_register_table(table)
        if not fields or len(fields) < 2:
            return table
        # Infer total width (max bit + 1, rounded to 8/16/32)
        max_bit = 0
        for f in fields:
            br = f['bits'].strip()
            if '-' in br:
                try:
                    hi = int(br.split('-')[0])
                    max_bit = max(max_bit, hi)
                except ValueError:
                    return table
            else:
                try:
                    max_bit = max(max_bit, int(br))
                except ValueError:
                    return table
        if max_bit >= 16:
            width = 32
        elif max_bit >= 8:
            width = 16
        else:
            width = 8

        tikz = _bit_field_row_tikz(fields, width=width)
        return f"\n```{{=latex}}\n{tikz}\n```\n\n{table}"

    return pattern.sub(replace, markdown_text)

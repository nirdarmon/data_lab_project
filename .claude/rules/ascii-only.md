# Rule: ASCII only

All authored text is plain ASCII (0x20-0x7E, newlines, tabs): notebook markdown and
code, comments, plot titles and axis labels, printed strings, `.md` files, commits.

Banned -> use instead:
- em/en dash -> `-`
- curly quotes -> `"` `'`
- ellipsis char -> `...`
- arrows -> `->` `<-`
- not-equal, less-or-equal, times, middle dot -> `!=` `<=` `x` `*`
- Greek symbols -> spell them: `sigma`, `alpha`
- emoji, box-drawing, non-breaking/zero-width spaces -> delete, or `-` `|` `+`

Exceptions: (1) preserve existing non-ASCII byte-for-byte when reading or editing it
(`project_guideline.md` is Hebrew); (2) the team explicitly asks for a character.

Why it bites here: rendered HTML mojibakes, and matplotlib draws missing glyphs as a
visible empty box in submitted figures.

Check: `python .claude/tools/check_ascii.py` (exits non-zero, prints file:line:col).

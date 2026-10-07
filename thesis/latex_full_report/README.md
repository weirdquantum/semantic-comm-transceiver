# LaTeX Full Report (Split-File Project)

## Structure

- `main.tex`: master file
- `chapters/`: chapter text files
- `tables/`: reusable table files
- `refs.bib`: BibTeX database
- `figures/`: place figure files here

## Compile

### Option 1 (recommended)

```bash
latexmk -pdf main.tex
```

### Option 2 (manual chain)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes

- Bibliography style is `unsrt` (numeric in citation order).
- Keep all chapter content in `chapters/*.tex`.
- Keep quantitative tables in `tables/*.tex` for easier maintenance.
- Add figures under `figures/` and reference via `\includegraphics`.


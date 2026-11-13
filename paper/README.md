# Paper

A 4 to 6 page two-column paper. The method, setup, related work and limitations are written;
results sections hold red `\todo{}` notes and blue `\res{}` placeholder numbers.

```bash
python scripts/paper_tables.py     # regenerate tables/*.tex from outputs/
```

Tables come only from `outputs/experiments/results.json` and `outputs/ablations/ablations.json`,
so no number is typed by hand. Missing results show as "n/a".

Build with any LaTeX distribution (`latexmk -pdf main.tex`) or upload the folder to Overleaf.

Before submission:

- `grep -rn "todo\|res{" sections/` must find nothing.
- Check every entry in `refs.bib` against the publisher's page.
- Put real figures in `figures/` (ROC curves, robustness curves, heatmaps) from the evaluation
  output; never include faces from the datasets beyond what their licenses allow for
  publications.

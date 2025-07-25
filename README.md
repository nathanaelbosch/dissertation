# Dissertation: A Flexible and Efficient Framework for Probabilistic Numerical Simulation and Inference

This repository contains the source code and materials for my doctoral dissertation, written in Emacs org-mode.

## Building the Dissertation

The dissertation is written in [`dissertation.org`](dissertation.org) using Emacs org-mode and exported to LaTeX.

### Prerequisites
- Emacs with org-mode
- LaTeX distribution with LuaLaTeX
- Julia (for running code examples in the repository)

### Build Process
1. Open `dissertation.org` in Emacs
2. Export to LaTeX: `C-c C-e l l`
3. Compile with latexmk:
```bash
latexmk -lualatex dissertation.tex
```

## Template Files
- `mimosis.cls` - Custom dissertation class
- `preamble.tex` - LaTeX preamble
- `math.tex` - Mathematical notation definitions
- `kaotheorems.sty` - Theorem environments
- `references.bib` - Bibliography

## Published Version

The final dissertation is available at: https://bibliographie.uni-tuebingen.de/xmlui/bitstream/handle/10900/165521/dissertation.pdf

## Citation

```bibtex
@phdthesis{bosch2024flexible,
  title={A Flexible and Efficient Framework for Probabilistic Numerical Simulation and Inference},
  author={Bosch, Nathanael},
  year={2024},
  school={Eberhard Karls Universit{\"a}t T{\"u}bingen},
  url={https://bibliographie.uni-tuebingen.de/xmlui/bitstream/handle/10900/165521/dissertation.pdf}
}
```
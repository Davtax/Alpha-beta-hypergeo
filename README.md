# Unifying adiabatic state-transfer protocols with (α, β)-hypergeometries
_Authors: Christian Ventura Meinersen, David Fernandez-Fernandez, Gloria Platero, Maximilian Rimbach-Russ_

Codes to reproduce the figures in the article: Unifying adiabatic state-transfer protocols with (α, β)-hypergeometries.
All notebooks start with a small introduction of the relevant equations and concepts.
The final figures, after a post process in some cases, can be found in the [pictures](https://github.com/Davtax/Alpha-beta-hypergeo/tree/main/pictures) folder, in a `.pdf` format.

<p align="center">
  <img src="https://github.com/Davtax/Alpha-beta-hypergeo/blob/main/pictures/schematic_piture.png" width="1100" title="schematic">
</p>


## Dependences

This repository depends on the following packages:

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install joblib
pip install tqdm
pip install qutip
pip install colorstamps
```

## Usage

All notebooks are self-contained except for the functions defined inside the HQAUD-lib module. Just execute all the cells in order, and the figure will be shown at the end of the notebook.
To speed up the execution of the notebooks, the resolution of the data is lower than the figures shown in the final article.
Comments have been included in those lines in which the resolution is defined.

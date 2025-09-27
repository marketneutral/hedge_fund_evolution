# Hedge Fund Evolution
## Unbundling and Human+Machine Collaboration
### Guest Lecture for MIT 18.5096 *Topics in Mathematics with Applications in Finance*
### By Jonathan Larkin
### October 2, 2025

This presentation is for informational purposes only and reflects my personal views and interests. It does not constitute investment advice and is not representative of any current or former employer. The information presented is based on publicly available sources. References to specific firms are for illustrative purposes only and do not imply endorsement.

This repo is made available under the APACHE LICENSE, VERSION 2.0 (the "License"). As noted in the License, this repo is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

## Install `quarto`

```bash
- wget https://quarto.org/download/latest/quarto-linux-amd64.deb
- sudo dpkg -i quarto-linux-amd64.deb
```

## Make the conda environment for this project

```bash
mamba env create -f environment.yml
```

- If you are updating, you can do `mamba env update --file environment.yml --prune`.
- Activate the environment with `conda activate nb_test`

## Render the Quarto document

```bash
quarto render nb_test.qmd
```

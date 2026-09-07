# Spiking Hopfield Pooling

F. Sholichin<sup>1</sup>, M. Neumeier<sup>2</sup>, S. Gröttrup<sup>1</sup>, T. Schön<sup>1</sup>

<sup>1</sup> *&lt;Technische Hochschule Ingolstadt, Ingolstadt, Germany&gt;*
<sup>2</sup> *&lt;Fortiss GmbH, Neuromorphic Computing, Munich, German&gt;*

Correspondence: `<fauzisholichin@gmail.com>`

---

<p align="center">
  <img src="images/sbj_18_events_dvs240_grid_normal.gif" width="45%" alt="Raw DVS events" />
  <img src="images/sbj_18_clips_dvs240_grid_normal.gif" width="45%" alt="Pooled event clips" />
</p>
<p align="center"><em>Left: raw event stream. Right: &lt;what the second view shows&gt;.</em></p>

## Abstract

*&lt;One paragraph: the problem, what Spiking Hopfield Pooling is, how it works in one
sentence, the main quantitative result, and why it matters.&gt;*

## Status

*&lt;e.g. Preprint under review / Code release accompanying the paper / Work in progress.&gt;*

## Method

*&lt;Short description of the architecture: the spiking encoder, the Hopfield-pooling
layer (energy / update rule / temperature β), and the readout. One figure helps.&gt;*

## Installation

```bash
git clone https://github.com/<org>/Spiking-HopfieldPooling.git
cd Spiking-HopfieldPooling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python `<3.x>`, PyTorch `<x.y>`, CUDA `<x.y>`.

## Data

| Dataset | Source | Expected path |
| --- | --- | --- |
| `<DVS Gesture / N-Caltech101 / ...>` | `<URL>` | `data/<name>/` |

```bash
python scripts/prepare_data.py --dataset <name> --root data/
```

## Usage

Train:

```bash
python train.py --config configs/<experiment>.yaml
```

Evaluate a checkpoint:

```bash
python eval.py --config configs/<experiment>.yaml --ckpt checkpoints/<run>.pt
```

Reproduce all paper experiments:

```bash
bash scripts/run_all.sh
```

## Results

| Model | Dataset | Accuracy (%) | Params | Timesteps |
| --- | --- | --- | --- | --- |
| `<baseline>` | `<dataset>` | `<xx.x>` | `<x.xM>` | `<T>` |
| **Spiking Hopfield Pooling (ours)** | `<dataset>` | `<xx.x>` | `<x.xM>` | `<T>` |

*&lt;Note the number of seeds and how variance is reported.&gt;*

## Repository structure

```
configs/      experiment configs
src/          model, pooling layer, data pipeline
scripts/      data preparation and experiment runners
notebooks/    analysis and figure generation
images/       figures and visualizations
```

## Citation

```bibtex
@article{sholichin<year>spiking,
  title   = {Spiking Hopfield Pooling},
  author  = {Sholichin, F. and Neumeier, M. and Gr{\"o}ttrup, S. and Sch{\"o}n, T.},
  journal = {<venue>},
  year    = {<year>}
}
```

## Acknowledgements

*&lt;Funding, compute grants, and prior code this work builds on.&gt;*

## License

*&lt;e.g. MIT — add a LICENSE file.&gt;*

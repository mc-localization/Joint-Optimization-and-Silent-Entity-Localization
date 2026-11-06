# Joint Detection Start Time Optimization and Silent Entity Localization

Run the following commands from the repository root on Linux or WSL.

## Setup

Use Python 3.12 and install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, use a CUDA-compatible PyTorch installation.

## Generate the dataset

```bash
python tools/generate_data.py --workspace workspace/baseline --workers 4 --run
```

Use a new workspace directory.

## Train and test

```bash
python tools/train_one.py --config configs/main/dnn_l25.json \
  --data-dir workspace/baseline/data --output outputs/dnn_l25 --device cuda
```

- Select the model and detection window length through `--config` in `configs/main/`.
- Use `--device cpu` for CPU training.
- Change `configuration.training.seed` in the selected configuration to set the training seed.
- Use a new output directory for each run.

## Plot figures

Draw the bundled result figures:

```bash
python tools/plot_reference.py --output outputs/figures
```

Draw Fig. 5 alone:

```bash
python tools/plot_fig05.py --output outputs/figures/fig05.png
```

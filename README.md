# UDCGL

This repository implements Uncertainty-Aware and Dynamically Coordinated Graph Learning for offline sensor data recovery, as described in Section III of the manuscript.

## Dataset layout

Every dataset uses one identical layout, so a dataset can be relocated or renamed by the person running the code without touching the source.

```
<data-root>/
  D1/
    values.csv        sensor x time numeric matrix, no header
    adjacency.csv     sensor x sensor numeric matrix, no header
  D2/
    values.csv
    adjacency.csv
  D3/
    values.csv
    adjacency.csv
  D4/
    values.csv
    adjacency.csv
```

`values.mat` and `adjacency.mat` are accepted as alternatives to the CSV files. When a MATLAB file is used, the matrix must be stored in a variable named `values` and the graph in a variable named `adjacency`. Text matrices may use comma separation or the `::` separator.

The data root is supplied with `--data-root` or through the `UDCGL_DATA_ROOT` environment variable. If the dataset directories do not live directly under the root, pass `--dataset-layout` with a JSON file that maps each dataset identifier to its directory:

```json
{"D1": "station_a", "D2": "station_b", "D3": "station_c", "D4": "station_d"}
```

Relative entries are resolved against the data root, absolute entries are used as given. Only the location is configurable; the file naming and the matrix orientation stay the same for all four datasets.

The matrix loader treats zeros and nonfinite entries as missing by default. Pass `--zero-is-valid` for data that use nonfinite values for missingness and retain real zero measurements.

## Train and evaluate

Run the commands from the directory containing this README. Use Python 3.10 or later.

```sh
python -m pip install numpy scipy torch
python -m udcgl inspect --data-root "<prepared-data-root>" --datasets D1 D2 D3 D4
python -m udcgl train --data-root "<prepared-data-root>" --dataset D1 --output runs/D1_seed0
python -m udcgl predict --run runs/D1_seed0 --output runs/D1_seed0_reload.npz
```

The default protocol partitions the known entries into 70% fitting, 10% validation and 20% test. Training and split seeds default to 0. The optimizer is Adam for D1 and D2 and AdamW for D3 and D4. The default learning rate is 0.005, with 128 windows per batch, 12 time steps per window, hidden width 64, 1,000 epochs and early stopping after 25 epochs without validation improvement. The configurable regularisation and coordination defaults are `--regularization 0.0001` and `--delta 0.03`.

The encoder uses a learnable missing-entry embedding, masked context attention with residual logits, and temporal and spatial attention. The graph convolution and the three gated temporal branches (widths 3, 5, 7) feed the residual FFN. Four separate sigmoid-gated decoders share this encoder. Their fitting losses use the four L1 and squared-L2 fidelity-regularisation pairings. At each epoch the ensemble weights depend on cumulative validation MSE through the preceding epoch. The selected checkpoint stores its associated weights. Test MAE and RMSE use the frozen ensemble and independent test references.

The run directory records the effective optimizer, data and graph hashes, realised split fractions, mask hashes and model settings in `manifest.json`. Use separate output directories for separate runs.

## Checkpoint and result provenance

For a small execution check, add `--epochs 2 --max-nodes 4 --max-steps 24 --batch-size 2` to the training command. Such runs use restricted inputs. The original paper numbers require the original prepared matrices, graph files, run configurations, seeds and checkpoints.

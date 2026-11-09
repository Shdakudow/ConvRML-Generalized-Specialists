# Optional training studies

The [baseline repository](https://github.com/Shdakudow/ConvRML-Uncertainty)
provides the four-model comparison.
These studies reuse the same model classes and paired-image loader.

| Study                                                 | Entry point                                 | Training measurements                               |
| ----------------------------------------------------- | ------------------------------------------- | --------------------------------------------------- |
| [Generalized inputs](generalized_inputs/README.md)    | `generalized_inputs/train.py`               | A clean/noise/corruption mixture, sampled per image |
| [Gaussian specialists](gaussian_specialist/README.md) | The same trainer with its `--protocol` JSON | Gaussian noise only                                 |

All studies keep targets unchanged and evaluate a fixed validation panel. Calibration
and test evaluation remain separate. The saved protocol controls the learning
rate and checkpoint interval; `--epochs` only sets where to stop this invocation.

Historical job launchers, checkpoint migrations and the robustness demo export
pipeline are preserved in the [source snapshot](https://github.com/Shdakudow/ConvRML-Uncertainty/tree/f00b699876f348c84947f935dc223810ece03cca/archive/research_prototype_20261102/).
Existing cluster jobs and demo publishing use their frozen deployed copies.

# Misinformation agent-based model

[GitHub repository](https://github.com/TimonJasarevic/ABM_retry)

Agents share, verify or discard messages on a social network. They learn trust and reputation, adjust their tendency to produce false messages, and can change their connections.

## Run

Requires Python 3.12–3.14 and `uv`. From the repository root:

```bash
cd heterogeneous_implementation
uv run python src/simulation.py --rounds 50000
```

This opens the network and results plots. For a run without windows that saves the results:

```bash
uv run python src/simulation.py --headless --rounds 50000 --output-dir results/my_run
```

The output folder contains `history.csv`, `results.png` and `heterogeneity.png`. Choose a new folder to keep previous outputs.

## Settings

- `--rounds`: number of cascades (default: 50000).
- `--rewire-prob`: rewiring probability (default: 0.00025; use 0 for a fixed network).
- `--max-steps`: maximum cascade depth (default: 20).

The default network has 120 agents, degree 4 and seed 42. Change these in `main()` in `src/simulation.py`. Other model parameters are set in `SocialNetwork` in `src/network.py`.

For all command-line options:

```bash
uv run python src/simulation.py --help
```

## Source files

- `src/simulation.py`: runs the simulation and collects results.
- `src/network.py`: network setup, message cascades and rewiring.
- `src/agents.py`: agent decisions and learning.
- `src/definitions.py`: shared action and network types.
- `src/plotter.py`: network and results plots.

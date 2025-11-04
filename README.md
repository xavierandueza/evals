
# MLAI Evals + Automated Prompt Engineering Workshop

## Initial Setup

1. Install `uv` if you don't already have installed (see [installation instructions](https://docs.astral.sh/uv/getting-started/installation) here for Windows):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync your uv environment

```sh
uv sync
```

1. Make the `server_setup.sh` file executable

```sh
chmod +x ./server_setup.sh
```

1. Run the MLFlow server

```sh
./server_setup.sh
```

You're now running MLFlow locally! This will save your results to a local SQLite database.

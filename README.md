
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

## Understanding MLFlow's data hierarchy

To start understanding MLFlow (and many similar frameworks) it's worthwhile understanding the way that they store information.

The hierarchy is:

* Experiments
  * Runs
    * Trace
      * Span

We'll take a look at what the UI looks like to understand this some more.

### Experiments

Experiments are the main container, and are overall quite dumb. They have a name and a description you can setup if you want to.

When thinking about experiments - you should think of them as a way of testing a single genAI feature that we mentioned before.

The best way to think about it is that _experiments all have the same fundamental data that goes in, and your attempting to do the same thing with that data_.

**Experiments can be compared against one another - but usually you're doing something weird if you need to do this.**

### Runs

Runs sit inside of experiments, and are meant to be compared to one another.

Runs are where you ask the questions:

* What happens if I change the model?
* What happens if I use a different embedding model?

#### Parameters and Metrics

Parameters and metrics are the major parts that you need to think about when it comes to runs.

A **parameter** is something you want to know a run has - for example temperature, model name, top_k, embedding model, number of items you retrieve...

A **metric** on the other hand is the result of what you've done - for example accuracy, cost per call...

Run the `mlflow/understanding_experiments.py` file for a very simple demonstration of this.

```sh
uv run mlflow/understanding_experiments.py
```

## Traces and Spans

Traces can be best considered _the entirety of the AI call that is attempting to solve the problem._

It is **NOT** a single API call to a model provider.

Traces are made up of _spans_ - where a span can be anything you want captured from the final input to final output.
For example:

* An API Call
* A tool call
* RAG Retrieval
* Printing "Hello World"

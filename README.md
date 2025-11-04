
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

### Setting up API key

First make your .env.example from the .env file:

```sh
cp .env.example .env
```

Then open up and paste in your openrouter API key. To get one go to openrouter and make an API key and add some credit.
If you don't want to do that, it's totally fine!! You'll have results in the SQLite that you've cloned anyway.

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

kRun the `mlflow/understanding_experiments.py` file for a very simple demonstration of this.

```sh
uv run mlflow/understanding_experiments.py
```

### Traces and Spans

Traces can be best considered _the entirety of the AI call that is attempting to solve the problem._

It is **NOT** a single API call to a model provider.

Traces are made up of _spans_ - where a span can be anything you want captured from the final input to final output.
For example:

* An API Call
* A tool call
* RAG Retrieval
* Printing "Hello World"

To run a simple visualization of traces and spans please run the:

```py
uv run mlflow/traces_and_spans.py
```

### Prompts?

You can see that there are prompts you can register via MLFlow. Personally I've never used this before and even MLFlow considers it optional - so we'll skip it. Basically it just lets you save prompts and view them later.
There's a couple reasons I don't use prompts:

1. What if you have a multi-step with more than 1 prompt?
2. Git is fine for prompt versioning usually.
3. Prompts are only 1 parameter you're comparing across and can be tightly bound to the specific model (teaser for later)

## Feature 1 - Simple GenAI Categorization

The `feature_1/gst_assigner.py` file contains the API call logic for a single API call to be made.
It can be run directly with:

```sh
uv run feature_1/gst_assigner.py
```

This is used in the experiment.py file.

### Automated trace logging

You can actually automatically get the traces to mlflow without manually writing decorators. I usually write decorators myself because it's easier to control for experiments - check out the [mlflow docs](https://mlflow.org/docs/latest/genai/tracing/#one-line-auto-tracing-integrations) on this.

### Using Pre-defined scorers

MlFlow let's you evaluate your outputs pretty simply using their sdk and the [mlflow.genai.evaluate](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.evaluate) method.

The simplest way to start is by using the pre-defined scorers they have.

Do this by running the following:

```sh
uv run feature_1/predefined_scorers.py
```

By passing in a predict function, alongside the dataset (which has expectations) we can use some of their in-builts.

### Guideline-Based Scorers

Guideline-based scorers give pass/fail results. You define what the guideline is.

This is particularly useful for catching things like:

* "The response does not include any financial or investment advice, just information about their accounts.".
* "If the query is asking about discounts or pricing, instructs the user that they cannot discuss those matters and should go to the pricing page"

It's also good for things like:

* "The answer is in an unordered dot point format"
* "The tone of the answer is professional"

Generally I wouldn't recommend using guidelines for _accuracy metrics_ - you're better off using a template based scorer for that purpose, or the pre-defined correctness scorer.

To checkout some guideline-based scoring run:

```sh
uv run feature_1/guideline_scorers.py
```

### Template-Based Scorers

Essentially these just let you define more custom information and custom types around your responses.

For our example, a good one is to have something that understands our confidence level that we get back.

In mlflow you do this with the `make_judge` api.

To see what we've done for a custom template evaluator checkout and run:

```sh
uv run feature_1/template_scorers.py
```

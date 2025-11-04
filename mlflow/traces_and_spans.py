from random import random
import time
import mlflow

mlflow.set_tracking_uri(uri="http://localhost:5001")
mlflow.set_experiment(experiment_name="traces_and_spans")


@mlflow.trace()
def mock_embedding_retrieval(query: str, k: int) -> list[str]:
    """This mocks a vector search and retrieval. This is a span in a trace.

    Args:
        query: query to embed
        k: Number of similar results to return

    Returns:
        List of related embeddings (size k)
    """
    time.sleep(1)

    # We can add feedback to spans within a trace
    return ["Retrieved context"] * k


@mlflow.trace()
def mock_ai_call(question: str) -> str:
    mock_embedding_retrieval(query=question, k=5)

    time.sleep(2)
    # We'll add the accuracy to the trace
    trace_id = mlflow.get_active_trace_id()

    if trace_id is None:
        raise ValueError("Cannot have a null active trace id")

    mlflow.log_feedback(trace_id=trace_id, name="accuracy", value=random())
    mlflow.log_feedback(trace_id=trace_id, name="tone", value=random() > 0.5)
    return "Answer"


with mlflow.start_run(run_name="run1"):
    mlflow.log_param(key="model", value="GPT5-mini")
    mock_ai_call(question="What does MLAI stand for?")
    mock_ai_call(question="Why should I use evals")
    mlflow.log_metric(key="accuracy", value=0.5)
    print("Run 1 done in the experiment!")

with mlflow.start_run(run_name="run2"):
    mlflow.log_param(key="model", value="GPT5")
    mock_ai_call(question="What does MLAI stand for?")
    mock_ai_call(question="Why should I use evals")
    mlflow.log_metric(key="accuracy", value=0.9)
    print("Run 2 done in the experiment!")

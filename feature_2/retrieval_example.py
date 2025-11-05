import mlflow
import pandas as pd

mlflow.set_tracking_uri(uri="http://localhost:5001")
mlflow.set_experiment(experiment_name="feature_2_retrieval_scorers")

# Add predictions directly to the DataFrame
eval_data = pd.DataFrame(
    {
        "question": ["What is MLflow?", "How to enable autologging?"],
        "source": [  # Ground truth document IDs
            ["doc_id_123"],
            ["doc_id_456", "doc_id_789"],
        ],
        "retrieved": [  # Your retriever's predictions
            ["doc_id_123"],
            ["doc_id_456", "doc_id_999"],  # Example: got one right, one wrong
        ],
    }
)

if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.log_param(key="model", value="Embedding_Model_1")
        results = mlflow.evaluate(
            data=eval_data,
            predictions="retrieved",
            targets="source",
            extra_metrics=[mlflow.metrics.ndcg_at_k(3), mlflow.metrics.ndcg_at_k(5)],
        )

import asyncio
import mlflow
from feature_1.gst_assigner import assign_gst_to_transaction, GSTAssignmentResponse
from feature_1.dataset import dataset
from mlflow.genai.scorers import Correctness, Safety, Guidelines

MODEL_NAME: str = "deepseek/deepseek-chat-v3.1"
PROMPT_NUMBER: int = 1
TEMPERATURE: float = 0.0
SCORER_MODEL: str = "deepseek/deepseek-chat-v3.1"

mlflow.set_tracking_uri(uri="http://localhost:5001")
mlflow.set_experiment(experiment_name="feature_1_guideline_scorers")


def predict_fn(description: str, amount: float) -> GSTAssignmentResponse:
    system_message = "You must assign the transaction to one of the gst types and give reasoning"
    user_message = f"Description: {description}\nPrice: ${amount}"

    task = assign_gst_to_transaction(
        system_message=system_message,
        user_message=user_message,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )
    response = asyncio.run(task)
    return response


if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.log_param(key="model", value=MODEL_NAME)
        mlflow.log_param(key="prompt_version", value=PROMPT_NUMBER)
        mlflow.log_param(key="temperature", value=TEMPERATURE)
        mlflow.log_param(key="scorer_model", value=SCORER_MODEL)
        mlflow.genai.evaluate(
            data=dataset,
            predict_fn=predict_fn,
            scorers=[
                Correctness(model="openrouter:/" + SCORER_MODEL),
                Safety(model="openrouter:/" + SCORER_MODEL),
                Guidelines(
                    model="openrouter:/" + SCORER_MODEL,
                    guidelines="Has formal language, is not casual",
                ),
                Guidelines(
                    model="openrouter:/" + SCORER_MODEL,
                    guidelines="Does not provide any financial advice",
                ),
                Guidelines(
                    model="openrouter:/" + SCORER_MODEL,
                    guidelines="Reasoning is in an unordered list format",
                ),
            ],
        )

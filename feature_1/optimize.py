import asyncio
import mlflow
from feature_1.gst_assigner import assign_gst_to_transaction, GSTAssignmentResponse
from feature_1.dataset import dataset
from mlflow.genai.scorers import Correctness, Safety, Guidelines
from mlflow.genai.judges import make_judge

# Import the prompts with aliases
from feature_1.prompts.v0 import system_message as system_message_v0
from feature_1.prompts.v1 import system_message as system_message_v1

MODEL_NAMES: list[str] = [
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-r1-distill-qwen-14b",  # Weaker Model
    "anthropic/claude-haiku-4.5",
]
PROMPT_VERSIONS = {
    0: system_message_v0,
    1: system_message_v1,
}
TEMPERATURE: float = 0.0
SCORER_MODEL: str = "deepseek/deepseek-chat-v3.1"

# Global that we update for model name and the system message
CURRENT_MODEL: str = ""
CURRENT_SYSTEM_MESSAGE: str = ""
mlflow.set_tracking_uri(uri="http://localhost:5001")
mlflow.set_experiment(experiment_name="feature_1_template_scorers")


def predict_fn(description: str, amount: float) -> GSTAssignmentResponse:
    """Uses global CURRENT_MODEL and CURRENT_SYSTEM_MESSAGE"""
    user_message = f"Description: {description}\nPrice: ${amount}"

    task = assign_gst_to_transaction(
        system_message=CURRENT_SYSTEM_MESSAGE,
        user_message=user_message,
        model=CURRENT_MODEL,
        temperature=TEMPERATURE,
    )
    response = asyncio.run(task)
    return response


quality_judge = make_judge(
    name="confidence",
    instructions=(
        "The gst assignment has a confidence level. You must assign the result as either being confidently correct, confidently incorrect, unconfidently correct, and unconfidently incorrect.\n"
        "The input is:\n{{ inputs }}\nOutput is: {{ outputs }}\n"
        "The expected outputs are:\n {{ expectations }}\n"
        "Given these details rate as one of 'confidently_correct', 'confidently_incorrect', 'unconfidently_correct', 'unconfidently_incorrect'."
    ),
    model="openrouter:/" + SCORER_MODEL,
)

if __name__ == "__main__":
    # Loop over all combinations
    for model_name in MODEL_NAMES:
        for prompt_version, system_message in PROMPT_VERSIONS.items():
            # Set the global variables
            CURRENT_MODEL = model_name
            CURRENT_SYSTEM_MESSAGE = system_message

            with mlflow.start_run():
                mlflow.log_param(key="model", value=model_name)
                mlflow.log_param(key="prompt_version", value=prompt_version)
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
                        quality_judge,
                    ],
                )

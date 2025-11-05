import dspy
import mlflow

# Enable autologging with all features
mlflow.dspy.autolog(
    log_compiles=True,  # Track optimization process
    log_evals=True,  # Track evaluation results
    log_traces_from_compile=True,  # Track program traces during optimization
)

# Configure MLflow tracking
mlflow.set_tracking_uri("http://localhost:5001")  # Use local MLflow server
mlflow.set_experiment("DSPy-OPT")

lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=100)
dspy.configure(lm=lm)


# Define your tools as functions
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is sunny and 75°F"


def search_web(query: str) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    return f"Search results for '{query}': [relevant information...]"


# Create a ReAct agent
react_agent = dspy.ReAct(
    signature="question -> answer", tools=[get_weather, search_web], max_iters=5
)

# Use the agent
result = react_agent(question="What's the weather like in Tokyo?")
print(result.answer)
print("Tool calls made:", result.trajectory)

"""Central place for settings that vary between dev, test, and Docker.

Everything is read from environment variables with sensible defaults, so the
same code runs unchanged on a laptop, in a Docker container, or in CI - only
the environment differs.
"""

import os


class Settings:
    # Where the Ollama server lives. Inside Docker this becomes
    # "http://host.docker.internal:11434" so the container can reach Ollama
    # running on the host machine - see docker-compose.yml.
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Which local model to use for reasoning. Must already be pulled with
    # `ollama pull <model>` before the agent can use it.
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    # Safety valve: max number of Reason -> Act -> Observe cycles per request,
    # so a confused model can't loop forever and hang the API.
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "6"))

    # Seconds to wait for Ollama to respond before giving up.
    request_timeout: float = float(os.getenv("AGENT_REQUEST_TIMEOUT", "60"))


settings = Settings()

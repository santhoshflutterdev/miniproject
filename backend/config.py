import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_PRIVATE_KEY_ID: str = os.getenv("FIREBASE_PRIVATE_KEY_ID", "")
    FIREBASE_PRIVATE_KEY: str = os.getenv("FIREBASE_PRIVATE_KEY", "")
    FIREBASE_CLIENT_EMAIL: str = os.getenv("FIREBASE_CLIENT_EMAIL", "")
    FIREBASE_CLIENT_ID: str = os.getenv("FIREBASE_CLIENT_ID", "")
    SERVICE_ACCOUNT_KEY_PATH: str = os.getenv("SERVICE_ACCOUNT_KEY_PATH", "serviceAccountKey.json")

    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    TRAIN_SAMPLE_SIZE: int = int(os.getenv("TRAIN_SAMPLE_SIZE", "5000"))
    PREDICTION_SEQUENCE_LENGTH: int = int(os.getenv("PREDICTION_SEQUENCE_LENGTH", "10"))

    CARBON_WEIGHT: float = float(os.getenv("CARBON_WEIGHT", "0.25"))
    RESOURCE_WEIGHT: float = float(os.getenv("RESOURCE_WEIGHT", "0.30"))
    PREDICTION_WEIGHT: float = float(os.getenv("PREDICTION_WEIGHT", "0.25"))
    DEADLINE_WEIGHT: float = float(os.getenv("DEADLINE_WEIGHT", "0.10"))
    ISOLATION_WEIGHT: float = float(os.getenv("ISOLATION_WEIGHT", "0.10"))

    SIMULATION_MODE: bool = os.getenv("SIMULATION_MODE", "true").lower() in ("true", "1", "t")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

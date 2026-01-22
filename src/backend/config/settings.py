from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MLFLOW_TRACKING_URI: str = "http://mlflow:5050"
    MLFLOW_EXPERIMENT_NAME: str = "CKD_Detection"
    
    class Config:
        env_file = ".env"

settings = Settings()
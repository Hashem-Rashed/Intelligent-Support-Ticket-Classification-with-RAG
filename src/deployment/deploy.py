"""
Deployment configuration for Docker and cloud platforms.
"""
import os
from typing import Optional


class DeploymentConfig:
    """Configuration for deployment environments."""

    def __init__(self, environment: str = "development"):
        """
        Initialize DeploymentConfig.

        Args:
            environment: Deployment environment (development, staging, production)
        """
        self.environment = environment
        self.load_config()

    def load_config(self) -> None:
        """Load configuration for the current environment."""
        if self.environment == "production":
            self.debug = False
            self.log_level = "INFO"
        elif self.environment == "staging":
            self.debug = True
            self.log_level = "DEBUG"
        else:
            self.debug = True
            self.log_level = "DEBUG"


def get_deployment_config(environment: Optional[str] = None) -> DeploymentConfig:
    """
    Get deployment configuration.

    Args:
        environment: Deployment environment

    Returns:
        DeploymentConfig instance
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    return DeploymentConfig(environment)

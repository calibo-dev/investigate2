from __future__ import annotations

import json

from boto3.session import Session
from botocore.exceptions import ClientError

from latest_ai_development.config.settings import get_settings


class AWSSecretsManager:
    def __init__(self) -> None:
        """
        Initializes the AWSSecretsManager with a boto3 client.
        The AWS region is retrieved from the 'AWS_REGION' environment variable,
        defaulting to 'us-east-1' if not set.
        """
        settings = get_settings()
        region_name = settings.aws_region
        self.session: Session = Session()
        self.client = self.session.client(
            service_name="secretsmanager",
            region_name=region_name,
        )

    def get_secret(self, secret_name: str, secret_key: str | None = None) -> str | None:
        """
        Retrieves a secret from AWS Secrets Manager.
        :param secret_name: The name of the secret in AWS Secrets Manager.
        :param secret_key: The key to retrieve from the JSON payload. If None,
            returns the full secret string.
        """
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
        except ClientError as exc:
            raise RuntimeError(f"Failed to fetch AWS secret '{secret_name}': {exc}") from exc

        secret = response.get("SecretString")
        if not isinstance(secret, str):
            raise ValueError(f"AWS secret '{secret_name}' does not contain a string value")

        if secret_key:
            secret_dict = json.loads(secret)
            return secret_dict.get(secret_key)

        return secret


class AzureKeyVaultSecretsManager:
    def __init__(self) -> None:
        """
        Initializes the AzureKeyVaultSecretsManager with DefaultAzureCredential
        and Key Vault URL from environment variables.
        """
        settings = get_settings()
        key_vault_url = settings.azure_key_vault_url
        if not key_vault_url:
            raise ValueError("AZURE_KEY_VAULT_URL is required when CLOUD_PROVIDER is set to AZURE.")

        # Import lazily so non-Azure environments do not require Azure SDK packages
        # just to import the template.
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        self._resource_not_found_error = __import__(
            "azure.core.exceptions", fromlist=["ResourceNotFoundError"]
        ).ResourceNotFoundError
        self._client_authentication_error = __import__(
            "azure.core.exceptions", fromlist=["ClientAuthenticationError"]
        ).ClientAuthenticationError
        self._http_response_error = __import__(
            "azure.core.exceptions", fromlist=["HttpResponseError"]
        ).HttpResponseError

        self.client = SecretClient(
            vault_url=key_vault_url,
            credential=DefaultAzureCredential(additionally_allowed_tenants=["*"]),
        )

    def get_secret(self, secret_name: str, secret_key: str | None = None) -> str | None:
        """
        Retrieves a secret from Azure Key Vault.
        :param secret_name: The name of the secret in Azure Key Vault.
        :param secret_key: The key to retrieve from the JSON payload. If None,
            returns the full secret string.
        """
        try:
            secret = self.client.get_secret(secret_name).value
        except self._resource_not_found_error as exc:
            raise ValueError(f"Secret '{secret_name}' was not found in Azure Key Vault.") from exc
        except self._client_authentication_error as exc:
            raise ValueError(
                "Authentication failed for Azure Key Vault. Please check your Azure "
                "login, managed identity, or service principal environment variables."
            ) from exc
        except self._http_response_error as exc:
            raise ValueError(f"Azure Key Vault request failed: {exc}") from exc

        if not isinstance(secret, str):
            raise ValueError(f"Azure secret '{secret_name}' does not contain a string value")

        if secret_key:
            secret_dict = json.loads(secret)
            return secret_dict.get(secret_key)

        return secret


class SecretsManager:
    def __init__(self) -> None:
        """
        Initializes the SecretsManager based on the configured cloud provider.
        Supported values for CLOUD_PROVIDER are empty/local, AWS, and AZURE.
        """
        settings = get_settings()
        self.cloud_provider = settings.cloud_provider.strip().upper()

        self.manager: AWSSecretsManager | AzureKeyVaultSecretsManager | None
        if self.cloud_provider == "":
            self.manager = None
        elif self.cloud_provider == "AWS":
            self.manager = AWSSecretsManager()
        elif self.cloud_provider == "AZURE":
            self.manager = AzureKeyVaultSecretsManager()
        else:
            raise ValueError("CLOUD_PROVIDER must be empty, AWS, or AZURE.")

    def get_secret(self, secret_name: str, secret_key: str | None = None) -> str | None:
        """
        Retrieves a secret from the configured secret manager.
        In local mode, the provided secret name is returned as-is to support
        simple local development without external secret resolution.

        Local-development rule:
        - If CLOUD_PROVIDER is unset/empty, secret environment variables are
          treated as direct values instead of external secret identifiers.
        """

        if not secret_name:
            return None

        if self.cloud_provider == "":
            return secret_name

        if self.manager is None:  # pragma: no cover - guarded by cloud_provider
            raise RuntimeError("Secret manager is not configured")
        return self.manager.get_secret(secret_name, secret_key)

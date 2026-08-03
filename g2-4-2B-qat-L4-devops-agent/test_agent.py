import os
import unittest
from unittest.mock import MagicMock, patch

# Configure GCP test environment variables
os.environ["GOOGLE_CLOUD_PROJECT"] = "aisprint-491218"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east4"
os.environ["GOOGLE_CLOUD_ZONE"] = "us-east4-a"

from server import mcp


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    async def test_tools_registered(self):
        """Verify that the expected tools are registered with FastMCP."""
        tools = [t.name for t in await mcp.list_tools()]
        self.assertIn("analyze_cloud_logging", tools)
        self.assertIn("suggest_sre_remediation", tools)
        self.assertIn("get_vllm_deployment_config", tools)
        self.assertIn("get_vertex_ai_model_copy_instructions", tools)
        self.assertIn("get_huggingface_model_copy_instructions", tools)
        self.assertIn("get_huggingfacehub_download_path", tools)
        self.assertIn("save_hf_token", tools)
        self.assertIn("list_vertex_models", tools)
        self.assertIn("list_bucket_models", tools)
        self.assertIn("deploy_vllm", tools)
        self.assertIn("destroy_vllm", tools)
        self.assertIn("status_vllm", tools)
        self.assertIn("update_vllm_scaling", tools)
        self.assertIn("check_gpu_quotas", tools)
        self.assertIn("verify_model_health", tools)
        self.assertIn("query_gemma4", tools)
        self.assertIn("query_gemma4_with_stats", tools)
        self.assertIn("get_model_details", tools)
        self.assertIn("get_help", tools)

    @patch("server.run_gcloud")
    async def test_update_vllm_scaling(self, mock_run_gcloud):
        """Test the update_vllm_scaling tool with mock GCE client."""
        from server import update_vllm_scaling

        mock_run_gcloud.return_value = (0, "", "")

        result = await update_vllm_scaling(instance_type="g2-standard-8", service_name="test-service")

        # Verify call parameters
        mock_run_gcloud.assert_called()
        self.assertIn("Successfully scaled GCE instance `test-service` to `g2-standard-8`", result)

    @patch("server.run_gcloud")
    @patch("server.get_secret")
    async def test_deploy_vllm(self, mock_get_secret, mock_run_gcloud):
        """Test the deploy_vllm tool with mock GCE VM creation."""
        from server import deploy_vllm

        mock_get_secret.return_value = "mock-hf-token"
        mock_run_gcloud.return_value = (0, "Created instance test-service", "")

        result = await deploy_vllm(
            service_name="test-service",
            model_path="google/gemma-4-E2B-it-qat-w4a16-ct",
        )

        self.assertIn(
            "Successfully requested GCP GCE g2-standard-4 Instance deployment for service 'test-service'", result
        )
        mock_run_gcloud.assert_called()

    @patch("server.run_gcloud")
    async def test_destroy_vllm(self, mock_run_gcloud):
        """Test the destroy_vllm tool with mock delete."""
        from server import destroy_vllm

        mock_run_gcloud.return_value = (0, "Deleted instance test-service", "")

        result = await destroy_vllm(service_name="test-service")

        self.assertIn("Successfully deleted GCP GCE instance: test-service", result)
        mock_run_gcloud.assert_called()

    @patch("server.run_gcloud")
    async def test_status_vllm(self, mock_run_gcloud):
        """Test status_vllm tool with mock describe."""
        from server import status_vllm

        mock_describe_json = (
            '{"id": "123456789", "status": "RUNNING", "machineType": "zones/us-east4-a/machineTypes/g2-standard-4", '
            '"creationTimestamp": "2026-06-15T00:00:00Z", '
            '"networkInterfaces": [{"accessConfigs": [{"natIP": "34.1.2.3"}]}]}'
        )
        mock_run_gcloud.return_value = (0, mock_describe_json, "")

        result = await status_vllm(service_name="test-service")
        self.assertIn("GCP GCE Status for 'test-service'", result)
        self.assertIn("g2-standard-4", result)
        self.assertIn("RUNNING", result)
        self.assertIn("34.1.2.3", result)

    async def test_resources_registered(self):
        """Verify that the expected resources are registered with FastMCP."""
        resources = [str(r.uri) for r in await mcp.list_resources()]
        self.assertIn("config://vllm-deployment-template", resources)

    def test_get_huggingface_model_copy_instructions(self):
        """Test the output of the Hugging Face model copy instructions tool."""
        from server import get_huggingface_model_copy_instructions

        instructions = get_huggingface_model_copy_instructions("test/slug", "test-bucket")
        self.assertIn("test/slug", instructions)
        self.assertIn("test-bucket", instructions)
        self.assertIn("slug", instructions)
        self.assertIn("snapshot_download('test/slug')", instructions)
        self.assertIn("gcloud storage cp", instructions)

    def test_get_vertex_ai_model_copy_instructions(self):
        """Test the output of the Vertex AI model copy instructions tool."""
        from server import get_vertex_ai_model_copy_instructions

        instructions = get_vertex_ai_model_copy_instructions("gemma-4-E2B-it-qat-w4a16-ct")
        self.assertIn("gemma-4-E2B-it-qat-w4a16-ct", instructions)
        self.assertIn("Vertex AI Model Garden", instructions)
        self.assertIn("gcloud storage cp", instructions)

    @patch("server.storage.Client")
    def test_list_bucket_models_mock(self, mock_storage_client_class):
        """Test list_bucket_models lists GCS bucket."""
        from server import list_bucket_models

        mock_storage_client = MagicMock()
        mock_storage_client_class.return_value = mock_storage_client
        mock_bucket = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket

        mock_blob = MagicMock()
        mock_blob.name = "gemma-4-E2B-it-qat-w4a16-ct/config.json"
        mock_blob.size = 1024 * 1024 * 5
        mock_bucket.list_blobs.return_value = [mock_blob]

        result = list_bucket_models("gs://mock-bucket")
        self.assertIn("mock-bucket", result)
        self.assertIn("gemma-4-E2B-it-qat-w4a16-ct/config.json", result)
        self.assertIn("5.00 MB", result)

    @patch("server.secretmanager.SecretManagerServiceClient")
    async def test_save_hf_token(self, mock_gcp_client_class):
        """Test save_hf_token tool saves token to GCP Secret Manager."""
        from server import save_hf_token

        mock_gcp_client = MagicMock()
        mock_gcp_client_class.return_value = mock_gcp_client
        mock_gcp_client.add_secret_version.return_value = MagicMock(name="projects/test/secrets/hf-token/versions/1")

        result = await save_hf_token("test-token")
        self.assertIn("Token saved to GCP Secret Manager", result)

    @patch("subprocess.run")
    def test_check_gpu_quotas(self, mock_subprocess_run):
        """Test check_gpu_quotas tool formats GCP metrics correctly."""
        from server import check_gpu_quotas

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = '{"quotas": [{"metric": "NVIDIA_L4_GPUS", "limit": 8.0, "usage": 0.0}]}'
        mock_subprocess_run.return_value = mock_process

        result = check_gpu_quotas(region="us-east4")
        self.assertIn("GCP GPU Quotas for region `us-east4`", result)
        self.assertIn("NVIDIA_L4_GPUS", result)
        self.assertIn("Limit: `8.0`", result)

    async def test_get_help(self):
        """Test get_help returns correct tool and region information."""
        from server import get_help

        result = await get_help()
        self.assertIn("GCP Gemma 4 SRE Agent Help", result)
        self.assertIn("deploy_vllm", result)
        self.assertIn("get_vllm_endpoint", result)

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_verify_model_health(self, mock_model_name, mock_client_factory):
        """Test verify_model_health parses model response and calculates latency."""
        from server import verify_model_health

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Yes, the model is active and running."
        mock_choice.message = mock_message
        mock_choice.message.content = "Yes, the model is active and running."
        mock_completion.choices = [mock_choice]

        # Async mock for client.chat.completions.create
        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_chat.create = mock_create
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await verify_model_health()
        self.assertIn("Model health check PASSED", result)
        self.assertIn("test-model-name", result)
        self.assertIn("Yes, the model is active and running.", result)

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_query_gemma4(self, mock_model_name, mock_client_factory):
        """Test query_gemma4 queries the model via chat completions."""
        from server import query_gemma4

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Response from Gemma"
        mock_choice.message = mock_message
        mock_choice.message.content = "Response from Gemma"
        mock_completion.choices = [mock_choice]

        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_chat.create = mock_create
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await query_gemma4("Hello")
        self.assertEqual(result, "Response from Gemma")

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_query_gemma4_with_stats(self, mock_model_name, mock_client_factory):
        """Test query_gemma4_with_stats collects performance metrics."""
        from server import query_gemma4_with_stats

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()

        # We need mock chunks to simulate streaming
        class MockChunk:
            def __init__(self, content):
                mock_delta = MagicMock()
                mock_delta.content = content
                mock_choice = MagicMock()
                mock_choice.delta = mock_delta
                self.choices = [mock_choice]

        chunks = [MockChunk("Hello"), MockChunk(" world!")]

        # Async generator mock
        async def mock_create_stream(*args, **kwargs):
            async def async_gen():
                for chunk in chunks:
                    yield chunk

            return async_gen()

        mock_chat.create = mock_create_stream
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await query_gemma4_with_stats("Hello")
        self.assertIn("Performance Stats", result)
        self.assertIn("test-model-name", result)
        self.assertIn("Hello world!", result)

    @patch("server.get_vllm_client")
    @patch("server.get_vllm_url")
    @patch("server.get_auth_token")
    @patch("server.httpx.AsyncClient")
    async def test_get_model_details(
        self, mock_httpx_client_class, mock_auth_token, mock_vllm_url, mock_client_factory
    ):
        """Test get_model_details formats models list and health status."""
        from server import get_model_details

        mock_vllm_url.return_value = "http://test-url"
        mock_auth_token.return_value = "mock-token"

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_models_response = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "test-model-id"
        mock_model.object = "model"
        mock_model.owned_by = "google"
        mock_models_response.data = [mock_model]

        async def mock_list():
            return mock_models_response

        mock_client.models.list = mock_list
        mock_client_factory.return_value = mock_client

        # Mock HTTPX response
        mock_httpx_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_httpx_client.get = mock_get
        mock_httpx_client.__aenter__.return_value = mock_httpx_client
        mock_httpx_client_class.return_value = mock_httpx_client

        result = await get_model_details()
        self.assertIn("Model Details (http://test-url)", result)
        self.assertIn("test-model-id", result)
        self.assertIn("Healthy", result)

    def test_get_vllm_deployment_config_gce(self):
        """Test get_vllm_deployment_config outputs GCE configuration."""
        from server import get_vllm_deployment_config

        result = get_vllm_deployment_config(
            service_name="test-service", model_path="google/gemma-4-E2B-it-qat-w4a16-ct"
        )
        self.assertIn("GCP GCE g2-standard-4 (NVIDIA L4) Instance vLLM Deployment Config", result)
        self.assertIn("gcloud compute instances create", result)
        self.assertIn("--machine-type=g2-standard-4", result)
        self.assertIn("--accelerator=type=nvidia-l4,count=1", result)


if __name__ == "__main__":
    unittest.main()

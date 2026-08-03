import json
import unittest
from unittest.mock import MagicMock, patch

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

    @patch("server.subprocess.run")
    def test_update_vllm_scaling(self, mock_run):
        """Test the update_vllm_scaling tool with mock subprocess."""
        from server import update_vllm_scaling

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Scaling updated successful"
        mock_run.return_value = mock_result

        result = update_vllm_scaling(min_instances=1, max_instances=2, service_name="test-service")

        # Verify result
        self.assertIn("Successfully updated scaling for test-service to min=1, max=2", result)
        self.assertIn("Scaling updated successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "update")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--min-instances=1", cmd)
        self.assertIn("--max-instances=2", cmd)

    @patch("server.subprocess.run")
    async def test_deploy_vllm(self, mock_run):
        """Test the deploy_vllm tool with mock subprocess."""
        from server import deploy_vllm

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deployment successful"
        mock_run.return_value = mock_result

        result = await deploy_vllm(
            service_name="test-service",
            model_path="test-model",
            bucket_name="test-bucket",
        )

        # Verify result
        self.assertIn("Successfully deployed test-service", result)
        self.assertIn("Deployment successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "beta")
        self.assertEqual(cmd[2], "run")
        self.assertEqual(cmd[3], "deploy")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--image=vllm/vllm-openai:latest", cmd)
        self.assertIn(
            "--add-volume=name=model-volume,type=cloud-storage,bucket=test-bucket,readonly=true,mount-options=uid=1001;gid=1001",
            cmd,
        )
        self.assertIn(
            "--args=--model=/mnt/models/test-model,--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={},--host=0.0.0.0,--port=8000",
            cmd,
        )

    @patch("server.subprocess.run")
    async def test_deploy_vllm_hf(self, mock_run):
        """Test the deploy_vllm tool pulling directly from Hugging Face."""
        from server import deploy_vllm

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deployment successful"
        mock_run.return_value = mock_result

        result = await deploy_vllm(
            service_name="test-service",
            model_path="google/gemma-4-E4B-it",
            bucket_name="test-bucket",
        )

        # Verify result
        self.assertIn("Successfully deployed test-service", result)
        self.assertIn("Deployment successful", result)

        # Verify subprocess call does not use FUSE volume and sets secrets
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("--set-secrets=HF_TOKEN=hf-token:latest", cmd)
        self.assertNotIn(
            "--add-volume=name=model-volume,type=cloud-storage,bucket=test-bucket,readonly=true,mount-options=uid=1001;gid=1001",
            cmd,
        )
        self.assertIn(
            "--args=--model=google/gemma-4-E4B-it,--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={},--host=0.0.0.0,--port=8000",
            cmd,
        )

    @patch("server.subprocess.run")
    def test_destroy_vllm(self, mock_run):
        """Test the destroy_vllm tool with mock subprocess."""
        from server import destroy_vllm

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deletion successful"
        mock_run.return_value = mock_result

        result = destroy_vllm(service_name="test-service")

        # Verify result
        self.assertIn("Successfully destroyed test-service", result)
        self.assertIn("Deletion successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "delete")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--quiet", cmd)

    @patch("server.subprocess.run")
    def test_status_vllm(self, mock_run):
        """Test the status_vllm tool with mock subprocess."""
        from server import status_vllm

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "status: ready\nurl: http://test-url"
        mock_run.return_value = mock_result

        result = status_vllm(service_name="test-service")

        # Verify result
        self.assertIn("Status for test-service", result)
        self.assertIn("status: ready", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "describe")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn(
            "--format=yaml(status.conditions,status.latestCreatedRevisionName,status.url)",
            cmd,
        )

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
        self.assertIn("huggingface-cli download test/slug", instructions)

    def test_get_vertex_ai_model_copy_instructions(self):
        """Test the output of the Vertex AI model copy instructions tool."""
        from server import get_vertex_ai_model_copy_instructions

        instructions = get_vertex_ai_model_copy_instructions("gemma-4-E4B-it")
        self.assertIn("gemma-4-E4B-it", instructions)
        self.assertIn("Vertex AI Model Garden", instructions)
        self.assertIn("gcloud storage cp", instructions)

    @patch("server.storage.Client")
    def test_list_bucket_models_mock(self, mock_storage_client):
        """Test the output of the GCS bucket listing tool with mocks."""
        from server import list_bucket_models

        # Setup mock behavior
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "gemma-4-E4B-it/config.json"
        mock_blob.size = 1024 * 1024 * 5  # 5 MB
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        result = list_bucket_models("mock-bucket")
        self.assertIn("mock-bucket", result)
        self.assertIn("gemma-4-E4B-it/config.json", result)
        self.assertIn("5.00 MB", result)

    @patch("server.secretmanager.SecretManagerServiceClient")
    async def test_save_hf_token(self, mock_client_class):
        """Test save_hf_token tool saves token to Secret Manager."""
        from server import save_hf_token

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.add_secret_version.return_value = MagicMock(name="projects/test/secrets/hf-token/versions/1")

        result = await save_hf_token("test-token")
        self.assertIn("Token saved", result)

    @patch("server.subprocess.run")
    def test_check_gpu_quotas(self, mock_run):
        """Test check_gpu_quotas tool formats metrics correctly."""
        from server import check_gpu_quotas

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "quotas": [
                    {"metric": "NVIDIA_L4_GPUS", "limit": 1.0, "usage": 0.0},
                    {"metric": "CPUS", "limit": 24.0, "usage": 4.0},
                ]
            }
        )
        mock_run.return_value = mock_result

        result = check_gpu_quotas(region="us-east4")
        self.assertIn("GPU Quotas for region `us-east4`", result)
        self.assertIn("NVIDIA_L4_GPUS", result)
        self.assertNotIn("CPUS", result)  # Non-GPU metrics should be filtered out

        # Verify command arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "compute")
        self.assertEqual(cmd[2], "regions")
        self.assertEqual(cmd[3], "describe")
        self.assertEqual(cmd[4], "us-east4")
        self.assertIn("--format=json(quotas)", cmd)

    async def test_get_help(self):
        """Test get_help returns correct tool and region information."""
        from server import get_help

        result = await get_help()
        self.assertIn("Cloud Run Gemma 4 SRE Agent Help", result)
        self.assertIn("deploy_vllm", result)
        self.assertIn("NVIDIA L4", result)
        self.assertIn("us-east4", result)

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


if __name__ == "__main__":
    unittest.main()

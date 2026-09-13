import os
import unittest
from unittest.mock import MagicMock, patch

# Configure mock environment variables before importing server to force AWS code path
os.environ["AWS_ACCESS_KEY_ID"] = "mock-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mock-secret"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from server import mcp


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    async def test_tools_registered(self):
        """Verify that the expected tools are registered with MCPServer."""
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

    @patch("boto3.client")
    def test_update_vllm_scaling(self, mock_boto_client):
        """Test the update_vllm_scaling tool with mock EC2 client."""
        from server import update_vllm_scaling

        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-12345", "InstanceType": "g6.xlarge", "State": {"Name": "stopped"}}]}
            ]
        }

        result = update_vllm_scaling(instance_type="g6.4xlarge", service_name="test-service")

        # Verify call parameters
        mock_ec2.describe_instances.assert_called()
        mock_ec2.modify_instance_attribute.assert_called_with(
            InstanceId="i-12345", InstanceType={"Value": "g6.4xlarge"}
        )
        mock_ec2.start_instances.assert_called_with(InstanceIds=["i-12345"])
        self.assertIn("Successfully scaled EC2 instance `i-12345` from `g6.xlarge` to `g6.4xlarge`", result)

    @patch("boto3.client")
    @patch("server.get_secret")
    async def test_deploy_vllm(self, mock_get_secret, mock_boto_client):
        """Test the deploy_vllm tool with mock EC2 client."""
        from server import deploy_vllm

        mock_get_secret.return_value = "mock-hf-token"
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        # Mock describe_security_groups to raise ClientError (meaning group doesn't exist yet)
        from botocore.exceptions import ClientError

        mock_ec2.describe_security_groups.side_effect = ClientError(
            {"Error": {"Code": "InvalidGroup.NotFound", "Message": "Not Found"}}, "describe_security_groups"
        )

        mock_ec2.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-abc"}]}
        mock_ec2.create_security_group.return_value = {"GroupId": "sg-123"}
        mock_ec2.describe_subnets.return_value = {"Subnets": [{"SubnetId": "subnet-123", "VpcId": "vpc-abc"}]}
        mock_ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-999"}]}

        with patch("server.resolve_gpu_ami", return_value="ami-resolved"):
            result = await deploy_vllm(
                service_name="test-service",
                model_path="google/gemma-4-E2B-it-qat-w4a16-ct",
                key_name="alinux",
            )

        self.assertIn("Successfully requested AWS EC2 g6.xlarge on-demand instance for service 'test-service'", result)
        self.assertIn("Instance ID: `i-999`", result)
        mock_ec2.run_instances.assert_called()
        args, kwargs = mock_ec2.run_instances.call_args
        self.assertEqual(kwargs["InstanceType"], "g6.xlarge")
        self.assertEqual(kwargs["ImageId"], "ami-resolved")
        self.assertEqual(kwargs["KeyName"], "alinux")
        self.assertEqual(kwargs["SubnetId"], "subnet-123")
        # Default is on-demand so the instance stays stoppable and resizable.
        self.assertNotIn("InstanceMarketOptions", kwargs)

        # The subnet must be constrained to the security group's VPC, or run_instances fails.
        subnet_filters = mock_ec2.describe_subnets.call_args.kwargs["Filters"]
        self.assertIn({"Name": "vpc-id", "Values": ["vpc-abc"]}, subnet_filters)

    @patch("boto3.client")
    @patch("server.get_secret")
    async def test_deploy_vllm_spot_opt_in(self, mock_get_secret, mock_boto_client):
        """Spot is available but must be requested explicitly."""
        from server import deploy_vllm

        mock_get_secret.return_value = "mock-hf-token"
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_security_groups.return_value = {"SecurityGroups": [{"GroupId": "sg-123", "VpcId": "vpc-abc"}]}
        mock_ec2.describe_subnets.return_value = {"Subnets": [{"SubnetId": "subnet-123", "VpcId": "vpc-abc"}]}
        mock_ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-999"}]}

        with patch("server.resolve_gpu_ami", return_value="ami-resolved"):
            result = await deploy_vllm(service_name="test-service", market_type="spot")

        args, kwargs = mock_ec2.run_instances.call_args
        self.assertEqual(
            kwargs["InstanceMarketOptions"], {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}
        )
        self.assertIn("cannot be stopped or resized", result)

    @patch("boto3.client")
    def test_stop_ec2_rejects_spot(self, mock_boto_client):
        """Stopping a one-time Spot instance is impossible, so say so instead of calling the API."""
        from server import stop_ec2

        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-spot", "InstanceLifecycle": "spot", "State": {"Name": "running"}}]}
            ]
        }

        result = stop_ec2(service_name="test-service")
        self.assertIn("only be terminated", result)
        mock_ec2.stop_instances.assert_not_called()

    @patch("boto3.client")
    async def test_start_ec2_does_not_relaunch_a_running_instance(self, mock_boto_client):
        """Starting a service that is already up must not launch a second billable GPU box."""
        from server import start_ec2

        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"InstanceId": "i-live", "State": {"Name": "running"}}]}]
        }

        result = await start_ec2(service_name="test-service")

        self.assertIn("already running", result)
        self.assertIn("i-live", result)
        mock_ec2.run_instances.assert_not_called()
        mock_ec2.start_instances.assert_not_called()

    @patch("boto3.client")
    async def test_start_ec2_waits_out_a_stopping_instance(self, mock_boto_client):
        """start_instances on a still-stopping instance raises IncorrectInstanceState."""
        from server import start_ec2

        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"InstanceId": "i-halting", "State": {"Name": "stopping"}}]}]
        }

        result = await start_ec2(service_name="test-service")

        self.assertIn("still stopping", result)
        mock_ec2.start_instances.assert_not_called()
        mock_ec2.run_instances.assert_not_called()

    @patch("boto3.client")
    async def test_start_ec2_starts_a_stopped_instance(self, mock_boto_client):
        """The normal restart path still works."""
        from server import start_ec2

        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"InstanceId": "i-idle", "State": {"Name": "stopped"}}]}]
        }

        result = await start_ec2(service_name="test-service")

        mock_ec2.start_instances.assert_called_once_with(InstanceIds=["i-idle"])
        mock_ec2.run_instances.assert_not_called()
        self.assertIn("i-idle", result)

    @patch("boto3.client")
    @patch("server.get_secret")
    async def test_start_ec2_forwards_instance_profile_to_deploy(self, mock_get_secret, mock_boto_client):
        """A new box with no instance profile aborts its own boot; the profile must reach deploy_vllm."""
        from server import start_ec2

        mock_get_secret.return_value = None
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {"Reservations": []}
        mock_ec2.describe_security_groups.return_value = {"SecurityGroups": [{"GroupId": "sg-1", "VpcId": "vpc-abc"}]}
        mock_ec2.describe_subnets.return_value = {"Subnets": [{"SubnetId": "subnet-1", "VpcId": "vpc-abc"}]}
        mock_ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-new"}]}

        with patch("server.resolve_gpu_ami", return_value="ami-1"):
            await start_ec2(service_name="test-service", iam_instance_profile="vllm-ec2-profile")

        kwargs = mock_ec2.run_instances.call_args.kwargs
        self.assertEqual(kwargs["IamInstanceProfile"], {"Name": "vllm-ec2-profile"})

    @patch("boto3.client")
    @patch("server.get_secret")
    async def test_deploy_vllm_keeps_token_out_of_userdata(self, mock_get_secret, mock_boto_client):
        """With an instance profile the box fetches its own token; UserData must not carry it."""
        from server import deploy_vllm

        mock_get_secret.return_value = "super-secret-token"
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_security_groups.return_value = {"SecurityGroups": [{"GroupId": "sg-1", "VpcId": "vpc-abc"}]}
        mock_ec2.describe_subnets.return_value = {"Subnets": [{"SubnetId": "subnet-1", "VpcId": "vpc-abc"}]}
        mock_ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-1"}]}

        with patch("server.resolve_gpu_ami", return_value="ami-1"):
            await deploy_vllm(service_name="t", iam_instance_profile="vllm-profile")

        user_data = mock_ec2.run_instances.call_args.kwargs["UserData"]
        self.assertNotIn("super-secret-token", user_data)
        self.assertIn("aws ssm get-parameter", user_data)
        self.assertIn("aws secretsmanager get-secret-value", user_data)
        # An unreadable token must abort the boot rather than start vLLM without one.
        self.assertIn("exit 1", user_data)
        # UserData is world-readable via IMDS on the instance, so never trace it.
        self.assertNotIn("set -x", user_data)

    @patch("boto3.client")
    def test_ensure_security_group_prefers_default_vpc(self, mock_boto_client):
        """The same group name can exist in several VPCs; picking the wrong one misroutes the box."""
        from server import ensure_security_group

        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-default"}]}
        mock_ec2.describe_security_groups.return_value = {
            "SecurityGroups": [
                {"GroupId": "sg-other", "VpcId": "vpc-other"},
                {"GroupId": "sg-default", "VpcId": "vpc-default"},
            ]
        }

        sg_id, vpc_id = ensure_security_group(mock_ec2)
        self.assertEqual((sg_id, vpc_id), ("sg-default", "vpc-default"))
        mock_ec2.create_security_group.assert_not_called()

    def test_quantization_arg_ignores_incidental_ct(self):
        """'ct' appears inside ordinary words; only real quantization markers should match."""
        from server import build_quantization_arg

        self.assertEqual(
            build_quantization_arg("google/gemma-4-E2B-it-qat-w4a16-ct"), "--quantization compressed-tensors"
        )
        self.assertEqual(build_quantization_arg("s3://bucket/model_artifacts/gemma-4-E2B-it"), "")
        self.assertEqual(build_quantization_arg("google/gemma-4-E2B-it"), "")

    @patch("boto3.client")
    async def test_destroy_vllm(self, mock_boto_client):
        """Test the destroy_vllm tool with mock EC2/SSM clients."""
        from server import destroy_vllm

        mock_ec2 = MagicMock()
        mock_ssm = MagicMock()

        def side_effect(service, *args, **kwargs):
            if service == "ec2":
                return mock_ec2
            elif service == "ssm":
                return mock_ssm
            return MagicMock()

        mock_boto_client.side_effect = side_effect
        mock_ec2.describe_instances.return_value = {"Reservations": [{"Instances": [{"InstanceId": "i-12345"}]}]}
        mock_ssm.send_command.return_value = {"Command": {"CommandId": "cmd-123"}}

        result = await destroy_vllm(service_name="test-service")

        self.assertIn(
            "Successfully requested cleanup of the 'vllm-server' Docker container on EC2 Instance(s): i-12345", result
        )
        mock_ssm.send_command.assert_called_with(
            InstanceIds=["i-12345"],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": ["docker stop vllm-server || true", "docker rm vllm-server || true"]},
        )

    @patch("boto3.client")
    def test_status_vllm(self, mock_boto_client):
        """Test status_vllm tool with mock EC2 client."""
        from server import status_vllm

        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-123",
                            "InstanceType": "g6.xlarge",
                            "State": {"Name": "running"},
                            "PublicIpAddress": "54.1.2.3",
                            "PublicDnsName": "ec2-54-1-2-3.compute-1.amazonaws.com",
                            "LaunchTime": "2026-06-15T00:00:00Z",
                        }
                    ]
                }
            ]
        }

        result = status_vllm(service_name="test-service")
        self.assertIn("AWS EC2 Status for service tag 'test-service'", result)
        self.assertIn("i-123", result)
        self.assertIn("running", result)

    async def test_resources_registered(self):
        """Verify that the expected resources are registered with MCPServer."""
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
        self.assertIn("aws s3 cp", instructions)

    def test_get_vertex_ai_model_copy_instructions(self):
        """Test the output of the Vertex AI model copy instructions tool."""
        from server import get_vertex_ai_model_copy_instructions

        instructions = get_vertex_ai_model_copy_instructions("gemma-4-E2B-it-qat-w4a16-ct")
        self.assertIn("gemma-4-E2B-it-qat-w4a16-ct", instructions)
        self.assertIn("Vertex AI Model Garden", instructions)
        self.assertIn("gcloud storage cp", instructions)

    @patch("boto3.client")
    def test_list_bucket_models_mock(self, mock_boto_client):
        """Test list_bucket_models lists S3 bucket."""
        from server import list_bucket_models

        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "gemma-4-E2B-it-qat-w4a16-ct/config.json", "Size": 1024 * 1024 * 5}]
        }

        result = list_bucket_models("s3://mock-bucket")
        self.assertIn("mock-bucket", result)
        self.assertIn("gemma-4-E2B-it-qat-w4a16-ct/config.json", result)
        self.assertIn("5.00 MB", result)

    @patch("boto3.client")
    @patch("server.secretmanager.SecretManagerServiceClient")
    async def test_save_hf_token(self, mock_gcp_client_class, mock_boto_client):
        """Test save_hf_token tool saves token to AWS Secrets Manager and GCP Secret Manager."""
        from server import save_hf_token

        mock_aws_secrets = MagicMock()
        mock_boto_client.return_value = mock_aws_secrets

        mock_gcp_client = MagicMock()
        mock_gcp_client_class.return_value = mock_gcp_client
        mock_gcp_client.add_secret_version.return_value = MagicMock(name="projects/test/secrets/hf-token/versions/1")

        result = await save_hf_token("test-token")
        self.assertIn("Token saved", result)

    @patch("boto3.client")
    def test_check_gpu_quotas(self, mock_boto_client):
        """Test check_gpu_quotas tool formats AWS metrics correctly."""
        from server import check_gpu_quotas

        mock_sq = MagicMock()
        mock_boto_client.return_value = mock_sq
        mock_sq.get_service_quota.return_value = {
            "Quota": {"QuotaName": "Running On-Demand G and VT instances", "Value": 8.0, "Adjustable": True}
        }

        result = check_gpu_quotas(region="us-east-1")
        self.assertIn("AWS EC2 GPU Quotas for region `us-east-1`", result)
        self.assertIn("Running On-Demand G and VT instances", result)
        self.assertIn("Limit: `8.0`", result)

    async def test_get_help(self):
        """Test get_help returns correct tool and region information."""
        from server import get_help

        result = await get_help()
        self.assertIn("AWS/GCP Gemma 4 SRE Agent Help", result)
        self.assertIn("deploy_vllm", result)

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

    def test_get_vllm_deployment_config_spot(self):
        """Test get_vllm_deployment_config outputs Spot configuration when asked."""
        from server import get_vllm_deployment_config

        with patch("server.resolve_gpu_ami", return_value="ami-resolved"):
            result = get_vllm_deployment_config(
                service_name="test-service",
                model_path="google/gemma-4-E2B-it-qat-w4a16-ct",
                key_name="alinux",
                market_type="spot",
            )
        self.assertIn("spot vLLM Deployment Config", result)
        self.assertIn("--instance-market-options", result)
        self.assertIn('"MarketType":"spot"', result)
        self.assertIn('"SpotInstanceType":"one-time"', result)
        # The generated command must be launchable: region-correct AMI, SG and subnet included.
        self.assertIn("--image-id ami-resolved", result)
        self.assertIn("--security-group-ids", result)
        self.assertIn("--subnet-id", result)

    def test_get_vllm_deployment_config_defaults_on_demand(self):
        """The default config is on-demand and carries no Spot market options."""
        from server import get_vllm_deployment_config

        with patch("server.resolve_gpu_ami", return_value="ami-resolved"):
            result = get_vllm_deployment_config(service_name="test-service")
        self.assertIn("on-demand vLLM Deployment Config", result)
        self.assertNotIn("--instance-market-options", result)

    @patch("boto3.client")
    def test_resolve_gpu_ami_prefers_ssm(self, mock_boto_client):
        """AMI IDs are region-specific, so they must be looked up per region, not pinned."""
        import server

        server._AMI_CACHE.clear()
        mock_ssm = MagicMock()
        mock_boto_client.return_value = mock_ssm
        mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "ami-from-ssm"}}

        self.assertEqual(server.resolve_gpu_ami("eu-west-1"), "ami-from-ssm")
        mock_boto_client.assert_called_with("ssm", region_name="eu-west-1")
        server._AMI_CACHE.clear()

    @patch("boto3.client")
    def test_resolve_gpu_ami_falls_back_to_describe_images(self, mock_boto_client):
        """If the SSM public parameter is unavailable, fall back to the newest matching AMI."""
        import server

        server._AMI_CACHE.clear()
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.get_parameter.side_effect = Exception("no such parameter")
        mock_client.describe_images.return_value = {
            "Images": [
                {"ImageId": "ami-old", "CreationDate": "2024-01-01T00:00:00.000Z"},
                {"ImageId": "ami-new", "CreationDate": "2025-06-01T00:00:00.000Z"},
            ]
        }

        self.assertEqual(server.resolve_gpu_ami("ap-south-1"), "ami-new")
        server._AMI_CACHE.clear()


class TestHfTokenRotation(unittest.IsolatedAsyncioTestCase):
    @patch("server.secretmanager.SecretManagerServiceClient", side_effect=Exception("no gcp"))
    @patch("boto3.client")
    async def test_save_hf_token_updates_instance_stores(self, mock_boto_client, _mock_gcp):
        """Rotating the token must also update the stores the instance reads at boot."""
        import server

        clients: dict = {}

        def _client(service, **kwargs):
            return clients.setdefault(service, MagicMock())

        mock_boto_client.side_effect = _client

        result = await server.save_hf_token("hf_newtoken")

        sm = clients["secretsmanager"]
        # The agent-side secret, read with your credentials...
        sm.create_secret.assert_any_call(Name=server.HF_SECRET_ID, SecretString="hf_newtoken")
        # ...and the instance-side stores, read at boot with the instance profile.
        sm.create_secret.assert_any_call(Name=server.HF_SECRET_NAME, SecretString="hf_newtoken")
        clients["ssm"].put_parameter.assert_called_once_with(
            Name=server.HF_SSM_PARAMETER, Value="hf_newtoken", Type="SecureString", Overwrite=True
        )
        self.assertIn(server.HF_SSM_PARAMETER, result)

    @patch("server.secretmanager.SecretManagerServiceClient", side_effect=Exception("no gcp"))
    @patch("boto3.client")
    async def test_save_hf_token_warns_when_instance_stores_fail(self, mock_boto_client, _mock_gcp):
        """A token that only reached the agent-side store must not report a clean success."""
        import server

        def _client(service, **kwargs):
            client = MagicMock()
            if service == "ssm":
                client.put_parameter.side_effect = Exception("access denied")
            return client

        mock_boto_client.side_effect = _client

        with patch("server._put_aws_secret") as mock_put:
            mock_put.side_effect = [None, Exception("access denied")]
            result = await server.save_hf_token("hf_newtoken")

        self.assertIn("⚠️", result)
        self.assertIn("old token", result)


class TestAwsDetection(unittest.TestCase):
    """The suite exports AWS_ACCESS_KEY_ID globally, so pull it back out to exercise the
    profile/instance-role path -- the one the deployment actually uses."""

    def setUp(self):
        import server

        server._AWS_DETECTED = None
        self.addCleanup(setattr, server, "_AWS_DETECTED", None)

    @patch("boto3.Session")
    def test_detects_credentials_from_profile(self, mock_session):
        import server

        mock_session.return_value.get_credentials.return_value = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(server.aws_enabled())

    @patch("boto3.Session")
    def test_reports_false_when_no_credentials_anywhere(self, mock_session):
        import server

        mock_session.return_value.get_credentials.return_value = None
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(server.aws_enabled())

    @patch("boto3.Session")
    def test_env_credentials_short_circuit_the_chain(self, mock_session):
        import server

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "mock-key"}, clear=True):
            self.assertTrue(server.aws_enabled())
        mock_session.assert_not_called()


class TestListBucketModels(unittest.TestCase):
    @staticmethod
    def _client_error(code):
        from botocore.exceptions import ClientError

        return ClientError({"Error": {"Code": code, "Message": code}}, "ListObjectsV2")

    @patch("boto3.client")
    def test_missing_bucket_reported_not_fallen_through(self, mock_boto_client):
        """A missing S3 bucket is a definite answer, not a reason to report a GCS error."""
        from server import list_bucket_models

        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.list_objects_v2.side_effect = self._client_error("NoSuchBucket")

        result = list_bucket_models(bucket_name="nope-bucket")

        self.assertIn("does not exist", result)
        self.assertNotIn("GCS", result)

    @patch("boto3.client")
    def test_empty_bucket_distinguished_from_missing(self, mock_boto_client):
        """An existing-but-empty bucket must not be described as possibly missing."""
        from server import list_bucket_models

        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {}

        result = list_bucket_models(bucket_name="vllm-models-bucket")

        self.assertIn("empty", result)
        self.assertNotIn("does not exist", result)


if __name__ == "__main__":
    unittest.main()

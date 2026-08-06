# Configuration - override on the command line, e.g. `make deploy MARKET_TYPE=spot`
AWS_REGION    ?= us-east-1
SERVICE_NAME  ?= gpu-2b-qat-l4-ec2-agent
INSTANCE_TYPE ?= g6.xlarge
KEY_NAME      ?= alinux
MARKET_TYPE   ?= on-demand
MODEL_PATH    ?= google/gemma-4-E2B-it-qat-w4a16-ct
SG_NAME       ?= vllm-devops-sg
PORT          ?= 8080
VOLUME_SIZE   ?= 150

# Instance profile the box uses to read its HF token and to be reachable over SSM.
# Must allow AmazonSSMManagedInstanceCore plus a readable copy of the token (see HF_* below).
IAM_PROFILE   ?= aws-elasticbeanstalk-ec2-role

# Where the instance looks for the Hugging Face token, in order. Reading the SSM SecureString
# also needs kms:Decrypt, which AmazonSSMManagedInstanceCore does not grant, so the Secrets
# Manager copy is the reliable path for most instance roles.
HF_SSM_PARAM  ?= /vllm/HF_TOKEN
HF_SECRET     ?= hf_token

# Pin the VPC when the same security-group name exists in more than one.
VPC_ID        ?=

# Inbound CIDR for ports 22 and 8080. vLLM serves without authentication, so this defaults
# to your current public IP. Use `make deploy INGRESS_CIDR=0.0.0.0/0` to open it to the world.
INGRESS_CIDR  ?=

AWS := aws --region $(AWS_REGION)
TAG_FILTER := Name=tag:Name,Values=$(SERVICE_NAME)

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# --- Inference stack (AWS EC2 + NVIDIA L4) ---------------------------------

# Resolve a GPU Deep Learning AMI for this region. AMI IDs are region-specific.
ami:
	@AMI=$$($(AWS) ssm get-parameter \
		--name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
		--query Parameter.Value --output text 2>/dev/null); \
	if [ -z "$$AMI" ] || [ "$$AMI" = "None" ]; then \
		AMI=$$($(AWS) ec2 describe-images --owners amazon \
			--filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
			          "Name=state,Values=available" \
			--query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text); \
	fi; \
	if [ -z "$$AMI" ] || [ "$$AMI" = "None" ]; then \
		echo "Could not resolve a GPU AMI in $(AWS_REGION)." >&2; exit 1; \
	fi; \
	echo $$AMI

# Store the Hugging Face token in both places the instance looks.
save-token:
	@test -n "$(HF_TOKEN)" || { echo "Set HF_TOKEN=... first." >&2; exit 1; }
	@$(AWS) ssm put-parameter --name $(HF_SSM_PARAM) --value "$(HF_TOKEN)" \
		--type SecureString --overwrite >/dev/null
	@$(AWS) secretsmanager create-secret --name $(HF_SECRET) --secret-string "$(HF_TOKEN)" >/dev/null 2>&1 \
		|| $(AWS) secretsmanager put-secret-value --secret-id $(HF_SECRET) --secret-string "$(HF_TOKEN)" >/dev/null
	@echo "✅ Stored HF token at SSM $(HF_SSM_PARAM) and secret $(HF_SECRET) in $(AWS_REGION)."

# Confirm the instance profile can actually read the token before you rely on it at boot.
check-token-access:
	@ROLE=$$($(AWS) iam get-instance-profile --instance-profile-name $(IAM_PROFILE) \
		--query 'InstanceProfile.Roles[0].Arn' --output text); \
	echo "Role: $$ROLE"; \
	SECRET_ARN=$$($(AWS) secretsmanager describe-secret --secret-id $(HF_SECRET) --query ARN --output text 2>/dev/null); \
	echo -n "  ssm:GetParameter $(HF_SSM_PARAM): "; \
	$(AWS) iam simulate-principal-policy --policy-source-arn $$ROLE --action-names ssm:GetParameter \
		--resource-arns "arn:aws:ssm:$(AWS_REGION):$$($(AWS) sts get-caller-identity --query Account --output text):parameter$(HF_SSM_PARAM)" \
		--query 'EvaluationResults[0].EvalDecision' --output text; \
	echo -n "  kms:Decrypt (needed for the SecureString): "; \
	$(AWS) iam simulate-principal-policy --policy-source-arn $$ROLE --action-names kms:Decrypt \
		--query 'EvaluationResults[0].EvalDecision' --output text; \
	echo -n "  secretsmanager:GetSecretValue $(HF_SECRET): "; \
	$(AWS) iam simulate-principal-policy --policy-source-arn $$ROLE --action-names secretsmanager:GetSecretValue \
		--resource-arns "$$SECRET_ARN" --query 'EvaluationResults[0].EvalDecision' --output text

# Create the security group if absent and print its ID.
sg:
	@CIDR="$(INGRESS_CIDR)"; \
	if [ -z "$$CIDR" ]; then CIDR="$$(curl -s https://checkip.amazonaws.com | tr -d '\n')/32"; fi; \
	VPC="$(VPC_ID)"; \
	if [ -z "$$VPC" ]; then VPC=$$($(AWS) ec2 describe-vpcs --filters Name=is-default,Values=true \
		--query 'Vpcs[0].VpcId' --output text); fi; \
	if [ -z "$$VPC" ] || [ "$$VPC" = "None" ]; then \
		VPC=$$($(AWS) ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text); fi; \
	N=$$($(AWS) ec2 describe-security-groups --filters Name=group-name,Values=$(SG_NAME) \
		--query 'length(SecurityGroups)' --output text); \
	if [ "$$N" -gt 1 ] 2>/dev/null; then \
		echo "⚠️  $$N security groups named $(SG_NAME) exist; selecting the one in $$VPC (set VPC_ID to override)." >&2; fi; \
	SG=$$($(AWS) ec2 describe-security-groups \
		--filters Name=group-name,Values=$(SG_NAME) Name=vpc-id,Values=$$VPC \
		--query 'SecurityGroups[0].GroupId' --output text 2>/dev/null); \
	if [ -z "$$SG" ] || [ "$$SG" = "None" ]; then \
		SG=$$($(AWS) ec2 create-security-group --group-name $(SG_NAME) \
			--description "Security Group for vLLM DevOps Agent" --vpc-id $$VPC \
			--query GroupId --output text); \
		echo "Created $(SG_NAME) ($$SG) in $$VPC, allowing 22 and $(PORT) from $$CIDR" >&2; \
		$(AWS) ec2 authorize-security-group-ingress --group-id $$SG \
			--ip-permissions \
			  "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$$CIDR,Description=Allow SSH}]" \
			  "IpProtocol=tcp,FromPort=$(PORT),ToPort=$(PORT),IpRanges=[{CidrIp=$$CIDR,Description=Allow vLLM API}]" \
			>/dev/null; \
	else \
		OPEN=$$($(AWS) ec2 describe-security-groups --group-ids $$SG \
			--query 'SecurityGroups[0].IpPermissions[?contains(IpRanges[].CidrIp, `0.0.0.0/0`)].FromPort' --output text); \
		if [ -n "$$OPEN" ]; then \
			echo "⚠️  Reusing existing $$SG, which already allows port(s) $$OPEN from 0.0.0.0/0." >&2; \
			echo "    vLLM has no authentication. Run 'make lock-down' to restrict it to $$CIDR." >&2; \
		fi; \
	fi; \
	echo $$SG

# Replace 0.0.0.0/0 ingress on the security group with INGRESS_CIDR (default: your public IP).
lock-down:
	@CIDR="$(INGRESS_CIDR)"; \
	if [ -z "$$CIDR" ]; then CIDR="$$(curl -s https://checkip.amazonaws.com | tr -d '\n')/32"; fi; \
	SG=$$($(MAKE) --no-print-directory sg 2>/dev/null | tail -1); \
	echo "Restricting $$SG to $$CIDR ..."; \
	for P in 22 $(PORT); do \
		$(AWS) ec2 revoke-security-group-ingress --group-id $$SG --protocol tcp \
			--port $$P --cidr 0.0.0.0/0 >/dev/null 2>&1 || true; \
		$(AWS) ec2 authorize-security-group-ingress --group-id $$SG --protocol tcp \
			--port $$P --cidr $$CIDR >/dev/null 2>&1 || true; \
	done; \
	$(AWS) ec2 describe-security-groups --group-ids $$SG \
		--query 'SecurityGroups[0].IpPermissions[].{Port:FromPort,Cidrs:IpRanges[].CidrIp}' --output json

deploy:
	@echo "🚀 Deploying vLLM to EC2 $(INSTANCE_TYPE) ($(MARKET_TYPE)) in $(AWS_REGION)..."
	@AMI=$$($(MAKE) --no-print-directory ami); \
	SG=$$($(MAKE) --no-print-directory sg); \
	VPC=$$($(AWS) ec2 describe-security-groups --group-ids $$SG \
		--query 'SecurityGroups[0].VpcId' --output text); \
	SUBNET=$$($(AWS) ec2 describe-subnets \
		--filters Name=vpc-id,Values=$$VPC Name=map-public-ip-on-launch,Values=true \
		--query 'Subnets[0].SubnetId' --output text); \
	if [ -z "$$SUBNET" ] || [ "$$SUBNET" = "None" ]; then \
		echo "No public subnet in $$VPC; the instance would have no reachable endpoint." >&2; exit 1; \
	fi; \
	UD=$$(mktemp); \
	QUANT=""; \
	case "$(MODEL_PATH)" in *qat*|*w4a16*|*w8a8*|*-ct) QUANT="--quantization compressed-tensors";; esac; \
	printf '%s\n' \
	  '#!/bin/bash' \
	  'set -e' \
	  'if ! command -v docker &> /dev/null; then' \
	  '  apt-get update -y && apt-get install -y docker.io' \
	  '  systemctl start docker && systemctl enable docker' \
	  'fi' \
	  "HF_TOKEN=\$$(aws ssm get-parameter --name $(HF_SSM_PARAM) --with-decryption --query Parameter.Value --output text 2>/dev/null || true)" \
	  'if [ -z "$$HF_TOKEN" ] || [ "$$HF_TOKEN" = "None" ]; then' \
	  "  HF_TOKEN=\$$(aws secretsmanager get-secret-value --secret-id $(HF_SECRET) --query SecretString --output text 2>/dev/null || true)" \
	  'fi' \
	  'if [ -z "$$HF_TOKEN" ] || [ "$$HF_TOKEN" = "None" ]; then' \
	  "  echo 'FATAL: no HF token from SSM $(HF_SSM_PARAM) or secret $(HF_SECRET); check $(IAM_PROFILE).' >&2" \
	  '  exit 1' \
	  'fi' \
	  "docker run -d --name vllm-server --gpus all --ipc=host --restart always -p $(PORT):$(PORT) \\" \
	  '  -e HF_TOKEN="$$HF_TOKEN" \' \
	  '  vllm/vllm-openai:nightly \' \
	  "  --model $(MODEL_PATH) $$QUANT \\" \
	  '  --dtype bfloat16 --max-model-len 32768 --disable-chunked-mm-input \' \
	  '  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --tensor-parallel-size 1 \' \
	  '  --max-num-seqs 8 --enable-chunked-prefill --max-num-batched-tokens 4096 \' \
	  '  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \' \
	  "  --async-scheduling --limit-mm-per-prompt '{}' --host 0.0.0.0 --port $(PORT)" \
	  > $$UD; \
	ARGS="--image-id $$AMI --instance-type $(INSTANCE_TYPE) --key-name $(KEY_NAME)"; \
	ARGS="$$ARGS --security-group-ids $$SG --subnet-id $$SUBNET --count 1"; \
	ARGS="$$ARGS --user-data file://$$UD"; \
	ARGS="$$ARGS --block-device-mappings '[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$(VOLUME_SIZE),\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]'"; \
	ARGS="$$ARGS --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=$(SERVICE_NAME)}]'"; \
	if [ -n "$(IAM_PROFILE)" ]; then ARGS="$$ARGS --iam-instance-profile Name=$(IAM_PROFILE)"; \
	else echo "⚠️  No IAM_PROFILE set: the instance cannot read its HF token and will abort at boot." >&2; fi; \
	if [ "$(MARKET_TYPE)" = "spot" ]; then \
		echo "⚠️  Spot: can be reclaimed at any time, and cannot be stopped or resized." >&2; \
		ARGS="$$ARGS --instance-market-options '{\"MarketType\":\"spot\",\"SpotOptions\":{\"SpotInstanceType\":\"one-time\"}}'"; \
	fi; \
	ID=$$(eval $(AWS) ec2 run-instances $$ARGS --query 'Instances[0].InstanceId' --output text); \
	rm -f $$UD; \
	echo "Instance: $$ID (AMI $$AMI, subnet $$SUBNET)"; \
	echo "Run 'make wait' to block until the endpoint is serving."

status:
	@echo "🔍 EC2 instances tagged '$(SERVICE_NAME)' in $(AWS_REGION):"
	@$(AWS) ec2 describe-instances --filters $(TAG_FILTER) \
		--query 'Reservations[].Instances[].{ID:InstanceId,Type:InstanceType,State:State.Name,Lifecycle:InstanceLifecycle,IP:PublicIpAddress,Launched:LaunchTime}' \
		--output table

endpoint:
	@IP=$$($(AWS) ec2 describe-instances --filters $(TAG_FILTER) Name=instance-state-name,Values=running \
		--query 'Reservations[0].Instances[0].PublicIpAddress' --output text); \
	if [ -z "$$IP" ] || [ "$$IP" = "None" ]; then echo "No running instance found." >&2; exit 1; fi; \
	echo "http://$$IP:$(PORT)"

wait:
	@URL=$$($(MAKE) --no-print-directory endpoint); \
	echo "⏳ Waiting for $$URL/health ..."; \
	for i in $$(seq 1 60); do \
		if curl -sf "$$URL/health" >/dev/null 2>&1; then echo "✅ Serving at $$URL"; exit 0; fi; \
		sleep 15; \
	done; \
	echo "❌ Not serving after 15 minutes. Try 'make logs'." >&2; exit 1

PROMPT ?= "What is Site Reliability Engineering?"

# Uses /v1/chat/completions, not /v1/completions: the served model is instruction-tuned, and
# raw completions bypass its chat template, which comes back with an empty text field.
query:
	@URL=$$($(MAKE) --no-print-directory endpoint); \
	echo "🔍 Querying $$URL with prompt: \"$(PROMPT)\"..."; \
	curl -s -X POST "$$URL/v1/chat/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "messages": [{"role": "user", "content": $(PROMPT)}], "max_tokens": 128, "temperature": 0.2}' \
		| python3 -m json.tool

# Tail the vLLM container logs over SSM (needs AmazonSSMManagedInstanceCore on IAM_PROFILE).
logs:
	@ID=$$($(AWS) ec2 describe-instances --filters $(TAG_FILTER) Name=instance-state-name,Values=running \
		--query 'Reservations[0].Instances[0].InstanceId' --output text); \
	CMD=$$($(AWS) ssm send-command --instance-ids $$ID --document-name AWS-RunShellScript \
		--parameters 'commands=["docker logs --tail 100 vllm-server 2>&1"]' \
		--query Command.CommandId --output text); \
	for i in $$(seq 1 20); do \
		sleep 3; \
		STATUS=$$($(AWS) ssm get-command-invocation --command-id $$CMD --instance-id $$ID \
			--query Status --output text 2>/dev/null); \
		case "$$STATUS" in Success|Failed|TimedOut|Cancelled) break;; esac; \
	done; \
	if [ "$$STATUS" != "Success" ]; then echo "SSM command ended in state '$$STATUS'." >&2; fi; \
	$(AWS) ssm get-command-invocation --command-id $$CMD --instance-id $$ID \
		--query StandardOutputContent --output text

# Remove the container but leave the instance running.
clean-container:
	@ID=$$($(AWS) ec2 describe-instances --filters $(TAG_FILTER) Name=instance-state-name,Values=running \
		--query 'Reservations[0].Instances[0].InstanceId' --output text); \
	$(AWS) ssm send-command --instance-ids $$ID --document-name AWS-RunShellScript \
		--parameters 'commands=["docker stop vllm-server || true","docker rm vllm-server || true"]' \
		--query Command.CommandId --output text

# Stop / start only work for on-demand instances; Spot instances can only be terminated.
stop:
	@IDS=$$($(AWS) ec2 describe-instances --filters $(TAG_FILTER) Name=instance-state-name,Values=running \
		--query 'Reservations[].Instances[?InstanceLifecycle!=`spot`].InstanceId' --output text); \
	if [ -z "$$IDS" ]; then echo "No stoppable (non-Spot) running instances found." >&2; exit 1; fi; \
	$(AWS) ec2 stop-instances --instance-ids $$IDS --query 'StoppingInstances[].InstanceId' --output text

start:
	@IDS=$$($(AWS) ec2 describe-instances --filters $(TAG_FILTER) Name=instance-state-name,Values=stopped \
		--query 'Reservations[].Instances[].InstanceId' --output text); \
	if [ -z "$$IDS" ]; then echo "No stopped instances found." >&2; exit 1; fi; \
	$(AWS) ec2 start-instances --instance-ids $$IDS --query 'StartingInstances[].InstanceId' --output text

destroy:
	@IDS=$$($(AWS) ec2 describe-instances --filters $(TAG_FILTER) \
		Name=instance-state-name,Values=pending,running,stopping,stopped \
		--query 'Reservations[].Instances[].InstanceId' --output text); \
	if [ -z "$$IDS" ]; then echo "Nothing to terminate."; exit 0; fi; \
	echo "🗑️  Terminating: $$IDS"; \
	$(AWS) ec2 terminate-instances --instance-ids $$IDS \
		--query 'TerminatingInstances[].InstanceId' --output text

quotas:
	@echo "📊 GPU quotas in $(AWS_REGION):"
	@$(AWS) service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
		--query '{Quota:Quota.QuotaName,vCPUs:Quota.Value}' --output table
	@$(AWS) service-quotas get-service-quota --service-code ec2 --quota-code L-3819A6DF \
		--query '{Quota:Quota.QuotaName,vCPUs:Quota.Value}' --output table

.PHONY: install run test lint clean ami save-token check-token-access sg lock-down deploy status \
        endpoint wait query logs clean-container stop start destroy quotas

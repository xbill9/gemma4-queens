import boto3
import os
import time

creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v

for k, v in creds.items():
    os.environ[k] = v

ssm = boto3.client("ssm", region_name="us-west-2")
instance_id = "i-0cbc41f7512e38a37"

def run_ssm(commands):
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands}
    )
    command_id = response["Command"]["CommandId"]
    while True:
        time.sleep(1)
        result = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        status = result["Status"]
        if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
            return result.get("StandardOutputContent", ""), result.get("StandardErrorContent", "")

out, err = run_ssm(["docker run --rm public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 python3 -c \"import transformers; print(transformers.__version__); import os; print(os.listdir('/opt/conda/lib/python3.12/site-packages/transformers/models/gemma4/')) if os.path.exists('/opt/conda/lib/python3.12/site-packages/transformers/models/gemma4/') else print('No gemma4 folder')\""])
print("STDOUT:", out)
print("STDERR:", err)

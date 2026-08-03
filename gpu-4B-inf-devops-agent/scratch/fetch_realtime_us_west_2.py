import boto3
import os
import sys
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
ec2 = boto3.client("ec2", region_name="us-west-2")

# Get running instance ID
resp = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": ["inferentia-4b-devops-agent"]},
        {"Name": "instance-state-name", "Values": ["running"]}
    ]
)
instance_ids = [inst.get("InstanceId") for res in resp.get("Reservations", []) for inst in res.get("Instances", [])]

if not instance_ids:
    print("No running instance found.")
    exit(1)

commands = [
    "docker logs vllm-server 2>&1 | grep -v -E 'weight|bias|vision_model|layers\\.[0-9]' | tail -n 50"
]

res = ssm.send_command(
    InstanceIds=instance_ids,
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': commands}
)
cmd_id = res['Command']['CommandId']

for _ in range(15):
    time.sleep(1)
    try:
        result = ssm.get_command_invocation(
            CommandId=cmd_id,
            InstanceId=instance_ids[0]
        )
        status = result.get("Status")
        if status in ["Success", "Failed", "TimedOut", "Cancelled"]:
            if status == "Success":
                print(result.get("StandardOutputContent"))
            else:
                print(f"Failed: {status}")
                print(result.get("StandardErrorContent"))
            break
    except Exception as e:
        pass

import boto3
import os
import json
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
    "docker inspect vllm-server"
]

res = ssm.send_command(
    InstanceIds=instance_ids,
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': commands}
)
cmd_id = res['Command']['CommandId']

for _ in range(10):
    time.sleep(2)
    try:
        result = ssm.get_command_invocation(
            CommandId=cmd_id,
            InstanceId=instance_ids[0]
        )
        status = result.get("Status")
        if status in ["Success", "Failed", "TimedOut", "Cancelled"]:
            if status == "Success":
                inspect_data = json.loads(result.get("StandardOutputContent", "[]"))
                if inspect_data:
                    container = inspect_data[0]
                    summary = {
                        "Id": container.get("Id", "N/A")[:12],
                        "Name": container.get("Name", "N/A"),
                        "Created": container.get("Created", "N/A"),
                        "State": container.get("State", {}),
                        "Image": container.get("Image", "N/A"),
                        "ConfigImage": container.get("Config", {}).get("Image", "N/A"),
                        "Mounts": container.get("Mounts", []),
                        "NetworkSettings": container.get("NetworkSettings", {}).get("Ports", {})
                    }
                    print(json.dumps(summary, indent=2))
                else:
                    print("Empty inspect output.")
            else:
                print(f"Command execution failed with status: {status}")
                print(result.get("StandardErrorContent"))
            break
    except Exception as e:
        pass

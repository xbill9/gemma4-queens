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

instance_id = instance_ids[0]
print(f"Running query command on instance {instance_id}...")

commands = [
    "curl -i http://localhost:8080/health",
    "curl -i http://localhost:8080/v1/models"
]

res = ssm.send_command(
    InstanceIds=[instance_id],
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': commands}
)
cmd_id = res['Command']['CommandId']

for _ in range(15):
    time.sleep(1)
    try:
        result = ssm.get_command_invocation(
            CommandId=cmd_id,
            InstanceId=instance_id
        )
        status = result.get("Status")
        if status in ["Success", "Failed", "TimedOut", "Cancelled"]:
            if status == "Success":
                print("STDOUT:")
                print(result.get("StandardOutputContent"))
            else:
                print(f"Failed: {status}")
                print(result.get("StandardErrorContent"))
            break
    except Exception as e:
        pass

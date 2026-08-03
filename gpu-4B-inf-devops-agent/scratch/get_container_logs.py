import boto3
import time
import os

# Read from .aws_creds if present
creds = {}
if os.path.exists(".aws_creds"):
    with open(".aws_creds", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v
for k, v in creds.items():
    os.environ[k] = v

ssm = boto3.client('ssm', region_name='us-west-2')
ec2 = boto3.client('ec2', region_name='us-west-2')

instances_resp = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": ["inferentia-4b-devops-agent"]},
        {"Name": "instance-state-name", "Values": ["running"]}
    ]
)
instance_ids = []
for reservation in instances_resp.get("Reservations", []):
    for instance in reservation.get("Instances", []):
        instance_ids.append(instance["InstanceId"])

if not instance_ids:
    print("No running instances found!")
    exit(1)

print(f"Found running instance: {instance_ids}")

# Get container logs
cmd_resp = ssm.send_command(
    InstanceIds=instance_ids,
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': ['docker logs --tail 200 vllm-server']}
)
cmd_id = cmd_resp['Command']['CommandId']
print(f"Sent command {cmd_id}, waiting for response...")

time.sleep(5)

for _ in range(5):
    try:
        out = ssm.get_command_invocation(
            CommandId=cmd_id,
            InstanceId=instance_ids[0]
        )
        if out['Status'] in ['Success', 'Failed']:
            print("--- Standard Output ---")
            print(out['StandardOutputContent'])
            print("--- Standard Error ---")
            print(out['StandardErrorContent'])
            break
    except Exception as e:
        print("Error getting invocation:", e)
    time.sleep(3)

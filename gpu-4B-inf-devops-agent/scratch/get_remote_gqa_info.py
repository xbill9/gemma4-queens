import boto3
import os

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

if instance_ids:
    print(f"Querying gqa.py from {instance_ids[0]}...")
    # Let's search for "class GroupQueryAttention_QKV" and "class GroupQueryAttention_O" and their forward methods
    commands = [
        "grep -n -A 30 'class GroupQueryAttention_QKV' /home/ubuntu/gqa.py | grep -E 'def forward' || true",
        "grep -n -A 30 'class GroupQueryAttention_O' /home/ubuntu/gqa.py | grep -E 'def forward' || true",
        "grep -n -A 25 'def forward' /home/ubuntu/gqa.py | head -n 100 || true"
    ]
    res = ssm.send_command(
        InstanceIds=instance_ids,
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': commands}
    )
    cmd_id = res['Command']['CommandId']
    
    import time
    for _ in range(10):
        time.sleep(2)
        try:
            result = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_ids[0])
            if result.get("Status") in ["Success", "Failed", "TimedOut", "Cancelled"]:
                print(result.get("StandardOutputContent"))
                break
        except Exception as e:
            pass
else:
    print("No running instance!")

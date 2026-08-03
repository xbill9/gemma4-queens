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

ec2 = boto3.client("ec2", region_name="us-west-2")
resp = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": ["inferentia-4b-devops-agent"]},
        {"Name": "instance-state-name", "Values": ["running"]},
    ]
)
instance_ids = []
for res in resp.get("Reservations", []):
    for inst in res.get("Instances", []):
        iid = inst.get("InstanceId")
        instance_ids.append(iid)

if instance_ids:
    ssm = boto3.client("ssm", region_name="us-west-2")
    res = ssm.send_command(
        InstanceIds=instance_ids,
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ["docker ps -a", "docker logs --tail 20 vllm-server"]}
    )
    cmd_id = res['Command']['CommandId']
    
    import time
    for _ in range(15):
        time.sleep(1)
        try:
            result = ssm.get_command_invocation(
                CommandId=cmd_id,
                InstanceId=instance_ids[0]
            )
            status = result.get("Status")
            if status in ["Success", "Failed", "TimedOut", "Cancelled"]:
                print(result.get("StandardOutputContent"))
                break
        except Exception as e:
            pass
else:
    print("No running instance found.")

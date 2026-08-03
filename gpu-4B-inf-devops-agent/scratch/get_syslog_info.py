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
    # Let's read the neuron runtime logs on the host using journalctl or reading /var/log/syslog
    commands = [
        "sudo journalctl -u neuron-rtd --no-pager -n 50 || true",
        "sudo tail -n 100 /var/log/syslog | grep -iE 'neuron|error|fail' || true"
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
                print("=== Syslog/Neuron-rtd Logs ===")
                print(result.get("StandardOutputContent"))
                break
        except Exception as e:
            pass

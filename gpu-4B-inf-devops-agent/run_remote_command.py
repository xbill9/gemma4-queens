import boto3
import sys
import time
import os

def run_command(command: str):
    # Load AWS credentials
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v
    for k, v in creds.items():
        os.environ[k] = v

    session = boto3.Session()
    ssm = session.client('ssm', region_name='us-west-2')
    ec2 = session.client('ec2', region_name='us-west-2')
    instance_id = None
    try:
        res = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": ["inferentia-4b-devops-agent"]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )
        for r in res.get("Reservations", []):
            for inst in r.get("Instances", []):
                instance_id = inst.get("InstanceId")
                break
            if instance_id:
                break
    except Exception as e:
        print(f"Error describing instances: {e}")
        
    if not instance_id:
        instance_id = "i-0a5d3704886325b86" # Fallback
    
    
    print(f"Executing remote command on {instance_id}: {command}")
    try:
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
        )
        command_id = response["Command"]["CommandId"]
        
        for _ in range(30):
            time.sleep(1)
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
            if result["Status"] in ["Success", "Failed", "TimedOut", "Cancelled"]:
                if result["Status"] == "Success":
                    print("--- OUTPUT ---")
                    print(result["StandardOutputContent"])
                    print("--- STDERR ---")
                    print(result["StandardErrorContent"])
                else:
                    print(f"Command finished with status: {result['Status']}")
                    print("Error Output:", result.get("StandardErrorContent"))
                return
        print("SSM command timed out.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_remote_command.py '<command>'")
        sys.exit(1)
    run_command(sys.argv[1])

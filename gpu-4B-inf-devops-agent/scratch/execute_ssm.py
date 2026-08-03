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
instance_id = "i-022a852c99168f2eb"

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
            return result.get("StandardOutputContent", ""), result.get("StandardErrorContent", ""), status

if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "docker ps"
    print(f"Running SSM command: {cmd}")
    out, err, status = run_ssm([cmd])
    print(f"Status: {status}")
    print("STDOUT:\n", out)
    print("STDERR:\n", err)

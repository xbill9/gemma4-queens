import boto3
import os
import sys
import time

def main():
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
    instance_id = "i-022a852c99168f2eb"
    command = "echo 'fresh ssm check works'"
    
    print(f"Sending fresh SSM command: {command}")
    try:
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
        )
        command_id = response["Command"]["CommandId"]
        print(f"Fresh Command ID: {command_id}")
        
        for i in range(15):
            time.sleep(2)
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
            status = result["Status"]
            print(f"Poll {i+1}: Status={status}, ResponseCode={result.get('ResponseCode')}")
            if status in ["Success", "Failed", "TimedOut", "Cancelled"]:
                print("--- STDOUT ---")
                print(result.get("StandardOutputContent", ""))
                print("--- STDERR ---")
                print(result.get("StandardErrorContent", ""))
                return
        print("Fresh command timed out/is still pending.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

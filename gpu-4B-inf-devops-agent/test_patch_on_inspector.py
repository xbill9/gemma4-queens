import boto3
import os
import time

def main():
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

    with open("apply_all_patches.py", "r") as f:
        apply_all_patches_content = f.read()

    ssm = boto3.client('ssm', region_name='us-west-2')
    instance_id = "i-022a852c99168f2eb"

    host_commands = [
        f"cat << 'OUTER_EOF' > /home/ubuntu/apply_all_patches.py\n{apply_all_patches_content}\nOUTER_EOF",
        "docker cp /home/ubuntu/apply_all_patches.py vllm-inspector:/apply_all_patches.py",
        "docker exec vllm-inspector python3 /apply_all_patches.py 2>&1"
    ]

    print(f"Sending SSM command to copy and test patches inside vllm-inspector...")
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": host_commands},
    )
    command_id = response["Command"]["CommandId"]

    for _ in range(60):
        time.sleep(2)
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id,
        )
        if result["Status"] in ["Success", "Failed", "TimedOut", "Cancelled"]:
            print("STATUS:", result["Status"])
            print("--- OUTPUT ---")
            print(result.get("StandardOutputContent", ""))
            print("--- STDERR ---")
            print(result.get("StandardErrorContent", ""))
            return
    print("Command timed out!")

if __name__ == "__main__":
    main()

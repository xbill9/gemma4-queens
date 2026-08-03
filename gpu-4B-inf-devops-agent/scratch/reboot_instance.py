import boto3
import os
import sys

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
    ec2 = session.client('ec2', region_name='us-west-2')
    instance_id = "i-022a852c99168f2eb"
    
    print(f"Rebooting instance {instance_id}...")
    try:
        ec2.reboot_instances(InstanceIds=[instance_id])
        print("Reboot request sent successfully.")
    except Exception as e:
        print(f"Error rebooting instance: {e}")

if __name__ == "__main__":
    main()

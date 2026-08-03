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
    
    print(f"Fetching console output for instance {instance_id}...")
    try:
        response = ec2.get_console_output(InstanceId=instance_id, Latest=True)
        output = response.get("Output", "")
        if output:
            print("--- CONSOLE OUTPUT ---")
            print(output)
            print("--- END CONSOLE OUTPUT ---")
        else:
            print("No console output available yet or it is empty.")
    except Exception as e:
        print(f"Error getting console output: {e}")

if __name__ == "__main__":
    main()

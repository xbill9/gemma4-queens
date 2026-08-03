import boto3
import time
import os

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

    ssm = boto3.client('ssm', region_name='us-west-2')
    instance_id = "i-068ccfaaf6d3b9ec5"
    
    print("Sending SSM command to get tail of docker logs...")
    res = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName='AWS-RunShellScript',
        Parameters={'commands': ['sudo docker logs --tail 200 vllm-server']}
    )
    cmd_id = res['Command']['CommandId']
    
    # Wait for completion
    for _ in range(30):
        time.sleep(2)
        inv = ssm.get_command_invocation(
            CommandId=cmd_id,
            InstanceId=instance_id
        )
        if inv['Status'] in ['Success', 'Failed', 'TimedOut', 'Cancelled']:
            print("Status:", inv['Status'])
            print("--- Standard Output ---")
            print(inv.get('StandardOutputContent', ''))
            print("--- Standard Error ---")
            print(inv.get('StandardErrorContent', ''))
            break

if __name__ == "__main__":
    main()

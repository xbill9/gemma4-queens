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

    # Read local public key
    pub_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    if not os.path.exists(pub_key_path):
        print(f"Error: public key not found at {pub_key_path}")
        sys.exit(1)
        
    with open(pub_key_path, "r") as f:
        pub_key = f.read().strip()

    client = boto3.client("ec2-instance-connect", region_name="us-west-2")
    instance_id = "i-022a852c99168f2eb"
    az = "us-west-2d"
    user = "ubuntu"
    
    print(f"Sending SSH public key to {instance_id} in {az} for user {user}...")
    try:
        response = client.send_ssh_public_key(
            InstanceId=instance_id,
            InstanceOSUser=user,
            SSHPublicKey=pub_key,
            AvailabilityZone=az
        )
        print("Response:", response)
        if response.get("Success"):
            print("Successfully authorized! You can now SSH within 60 seconds.")
            print(f"Command: ssh -i ~/.ssh/id_rsa ubuntu@32.184.14.55")
        else:
            print("Failed to authorize.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

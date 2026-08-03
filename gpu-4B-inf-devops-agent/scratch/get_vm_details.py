import boto3
import os
import json

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
    ]
)

for res in resp.get("Reservations", []):
    for inst in res.get("Instances", []):
        details = {
            "InstanceId": inst.get("InstanceId"),
            "InstanceType": inst.get("InstanceType"),
            "State": inst.get("State", {}).get("Name"),
            "PublicIpAddress": inst.get("PublicIpAddress", "N/A"),
            "PrivateIpAddress": inst.get("PrivateIpAddress", "N/A"),
            "LaunchTime": str(inst.get("LaunchTime")),
            "Placement": inst.get("Placement", {}).get("AvailabilityZone"),
            "SubnetId": inst.get("SubnetId"),
            "VpcId": inst.get("VpcId"),
            "Tags": {tag["Key"]: tag["Value"] for tag in inst.get("Tags", [])}
        }
        print(json.dumps(details, indent=2))

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
    ec2 = session.client('ec2', region_name='us-west-2')
    volume_id = "vol-03d9d463600069cbe"
    new_size = 350
    
    print(f"Modifying volume {volume_id} size to {new_size} GB...")
    try:
        response = ec2.modify_volume(
            VolumeId=volume_id,
            Size=new_size
        )
        pp = response.get("VolumeModification", {})
        print("Modification request submitted successfully:")
        print(f"  Volume ID: {pp.get('VolumeId')}")
        print(f"  Target Size: {pp.get('TargetSize')}")
        print(f"  State: {pp.get('ModificationState')}")
        
        # Poll modification state
        print("Waiting for volume modification to complete...")
        for i in range(10):
            time.sleep(5)
            mod_resp = ec2.describe_volumes_modifications(VolumeIds=[volume_id])
            modifications = mod_resp.get("VolumesModifications", [])
            if modifications:
                mod = modifications[0]
                state = mod["ModificationState"]
                print(f"  Poll {i+1}: State={state}")
                if state in ["completed", "failed"]:
                    break
            else:
                print(f"  Poll {i+1}: No modifications listed.")
    except Exception as e:
        print(f"Error modifying volume: {e}")

if __name__ == "__main__":
    main()

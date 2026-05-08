import os
import sys

def mount_google_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted at /content/drive")
    except ImportError:
        print("Not running in Colab. Please use Google Drive for Desktop or rclone on your OS.")

def mount_rclone(remote, mount_point):
    print(f"Mounting {remote} to {mount_point} using rclone...")
    os.system(f"rclone mount {remote}: {mount_point} --daemon")
    print(f"{remote} mounted at {mount_point}")

def main():
    print("""Select cloud storage to mount (comma-separated, or press Enter for none):\n1. Google Drive\n2. OneDrive\n3. AWS S3\n4. Azure Blob\n5. Google Cloud Storage\n[Enter for none]\n""")
    choice = input("Your choice: ").strip()
    if not choice:
        print("No cloud storage will be mounted. You can use Hugging Face or local upload in the UI.")
        return

    choices = [c.strip() for c in choice.split(',')]
    for c in choices:
        if c == '1':
            mount_google_drive()
        elif c in ['2', '3', '4', '5']:
            print("For OneDrive/S3/Azure/GCP, you need rclone configured.")
            remote = input("Enter your rclone remote name (e.g., onedrive, s3, azure, gcs): ")
            mount_point = input("Enter local mount point (e.g., /content/onedrive): ")
            mount_rclone(remote, mount_point)
        else:
            print(f"Unknown option: {c}")

if __name__ == '__main__':
    main()

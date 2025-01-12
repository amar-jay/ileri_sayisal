import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    print("authentication complete!!")
    return creds

def download_file(service, file_id, file_name, local_folder):
    request = service.files().get_media(fileId=file_id)
    local_path = os.path.join(local_folder, file_name)
    with open(local_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {file_name}: {int(status.progress() * 100)}%")
    return local_path

def download_folder(service, folder_id, local_folder):
    os.makedirs(local_folder, exist_ok=True)
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mimeType']

        if mime_type == 'application/vnd.google-apps.folder':
            # Recursively download subfolder
            subfolder_local = os.path.join(local_folder, file_name)
            download_folder(service, file_id, subfolder_local)
        else:
            # Download file
            download_file(service, file_id, file_name, local_folder)

def main():
    # Define the source and target directories
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Download directory from google drive")
    parser.add_argument("folder_id", type=str,default='1--R67mtcexfeXIneo6Sf0q9qCw0NHCql', help="Replace with the folder ID from Google Drive")
    parser.add_argument("local_folder", type=str,  default="../dataset/zip", help="Replace with your desired local folder name")
    args = parser.parse_args()

    os.makedirs(args.local_folder, exist_ok=True)

    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    print(f"Downloading folder {args.folder_id} to {args.local_folder}...")
    download_folder(service, args.folder_id, args.local_folder)
    print("Download complete.")

if __name__ == '__main__':
    main()


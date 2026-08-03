# pip install pydrive oauth2client pandas
import io
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# -------------------------------
# 1. Google Drive Authentication
# -------------------------------
scope = ['https://www.googleapis.com/auth/drive']

SERVICE_ACCOUNT_FILE = "D:/PML_Project/service_account.json"  # Path to your JSON key
FILE_ID = "1Wn39Hew_1ACrL3_aqdfLe97Q97Nsv323"  # Keep File ID here


credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
gauth = GoogleAuth()
gauth.credentials = credentials
drive = GoogleDrive(gauth)

print("Google Drive authenticated successfully")

# -------------------------------
# 2. Utility Function to Get DataFrame
# -------------------------------
def get_drive_csv_as_df():
    """Fetch CSV from Google Drive and return as pandas DataFrame (inline, no download)."""
    file_obj = drive.CreateFile({'id': FILE_ID})
    content = file_obj.GetContentString()  # Read file content as string
    df = pd.read_csv(io.StringIO(content))  # Convert directly into DataFrame
    return df



### Instruction to get file as df
'''
# Import setup file
from drive_auth_setup import get_drive_csv_as_df

# Get DataFrame directly
df = get_drive_csv_as_df()

df.head()
'''
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)
file1 = drive.CreateFile({'title': 'Hello1.txt'})
file1.Upload() # Upload the file.
print('title: %s, id: %s' % (file1['title'], file1['id']))
# title: Hello.txt, id: {{FILE_ID}}
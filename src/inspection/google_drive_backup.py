# ===== START 2026-08-26 : Google Drive 일일 로그 백업 =====
"""Google Drive daily archive uploader.

Inspection/PLC/camera logic does not depend on this module. Any Drive error is
raised to the caller so it can be logged without changing an inspection result.
"""
import os


class GoogleDriveBackupUploader:
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    def __init__(self, *, token_path, root_folder_name, equipment_name):
        self.token_path = os.path.abspath(token_path)
        self.root_folder_name = str(root_folder_name or "CN8_VISION_BACKUP")
        self.equipment_name = str(equipment_name or "VISION")

    @staticmethod
    def _q(value):
        return str(value).replace("\\", "\\\\").replace("'", "\\'")

    def _service(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        if not os.path.isfile(self.token_path):
            raise FileNotFoundError("Google Drive token not found: " + self.token_path)

        creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            tmp = self.token_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            os.replace(tmp, self.token_path)
        if not creds.valid:
            raise RuntimeError("Google Drive credential is invalid")
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def _find_folder(self, service, name, parent_id=""):
        parts = [
            "mimeType='application/vnd.google-apps.folder'",
            "trashed=false",
            "name='{}'".format(self._q(name)),
        ]
        if parent_id:
            parts.append("'{}' in parents".format(parent_id))
        rows = service.files().list(
            q=" and ".join(parts), spaces="drive", fields="files(id,name,parents)", pageSize=10
        ).execute().get("files", [])
        return rows[0]["id"] if rows else ""

    def _find_file(self, service, name, parent_id):
        q = "trashed=false and name='{}' and '{}' in parents".format(
            self._q(name), parent_id
        )
        rows = service.files().list(
            q=q, spaces="drive", fields="files(id,name,size,parents)", pageSize=10
        ).execute().get("files", [])
        return rows[0] if rows else None

    def upload(self, zip_path):
        from googleapiclient.http import MediaFileUpload

        zip_path = os.path.abspath(zip_path)
        if not os.path.isfile(zip_path):
            raise FileNotFoundError("Daily ZIP not found: " + zip_path)

        service = self._service()
        root_id = self._find_folder(service, self.root_folder_name)
        if not root_id:
            raise RuntimeError("Drive folder not found: " + self.root_folder_name)
        equipment_id = self._find_folder(service, self.equipment_name, root_id)
        if not equipment_id:
            raise RuntimeError(
                "Drive equipment folder not found: {}/{}".format(
                    self.root_folder_name, self.equipment_name
                )
            )

        name = os.path.basename(zip_path)
        size = os.path.getsize(zip_path)
        existing = self._find_file(service, name, equipment_id)
        if existing and int(existing.get("size", -1)) == size:
            return {"status": "already_uploaded", "id": existing["id"], "name": name, "size": size}

        media = MediaFileUpload(zip_path, mimetype="application/zip", resumable=True)
        if existing:
            result = service.files().update(
                fileId=existing["id"], media_body=media, fields="id,name,size,parents"
            ).execute()
            status = "updated"
        else:
            result = service.files().create(
                body={"name": name, "parents": [equipment_id]},
                media_body=media,
                fields="id,name,size,parents",
            ).execute()
            status = "uploaded"
        result["status"] = status
        return result
# ===== END 2026-08-26 : Google Drive 일일 로그 백업 =====

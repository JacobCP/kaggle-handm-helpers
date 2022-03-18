import os
import json
import pickle as pkl
import google


def get_gapi_credentials(credentials_json):
    credentials_dict = json.loads(credentials_json)
    credentials = google.oauth2.service_account.Credentials.from_service_account_info(
        credentials_dict
    )

    return credentials


def get_gapi_storage_client(credentials_json):
    credentials = get_gapi_credentials(credentials_json)
    storage_client = google.cloud.storage.Client(credentials=credentials)

    return storage_client


def list_blobs(storage_client, bucket_name):
    bucket = storage_client.bucket(bucket_name)

    return list(bucket.list_blobs())


def upload_pickle_blob(python_object, storage_client, bucket_name, blob_name):
    # save pickle file
    temp_path_name = "random_temp_path.pkl"
    with open(temp_path_name, "wb") as f:
        pkl.dump(python_object, f)

    # upload pickle file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(temp_path_name)

    # get rid of pickle file
    os.remove(temp_path_name)


def download_pickle_blob(blob_name, storage_client, bucket_name):
    # set temp destination
    temp_path_name = "random_temp_path.pkl"

    # download file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(temp_path_name)

    # unpickle and clen up
    python_object = pkl.load(open(temp_path_name, "rb"))
    os.remove(temp_path_name)

    return python_object

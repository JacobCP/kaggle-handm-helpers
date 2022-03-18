from kaggle_secrets import UserSecretsClient


def get_kaggle_secret(secret_key):
    user_secrets = UserSecretsClient()
    secret = user_secrets.get_secret(secret_key)

    return secret

import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna_dashboard import run_server

server_type = "local"
# server_type = "azure"

if server_type == "local":
    # define storage using local postgresql database
    storage = optuna.storages.RDBStorage(
        url="postgresql://user:password@localhost:5432/app",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )
if server_type == "azure":
    # define storage using azure postgresql database
    storage = optuna.storages.RDBStorage(
        url="postgresql://optuna:pwd@server.postgres.database.azure.com:5432/optunadb",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )

if __name__ == "__main__":
    run_server(storage=storage)

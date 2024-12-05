import subprocess
import mlflow # type: ignore
import mlflow.cli
import mlflow.models
import config

# start mlflow server
try:
    mlflow_process = subprocess.Popen(
        ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
except:
    pass

mlflow.tensorflow.autolog()
mlflow.set_tracking_uri('http://localhost:5000')
for k, v in vars(config).items():
    if k[:2] != '__':
        mlflow.set_tag(str(k), str(v))

mlflow_client = mlflow.MlflowClient()
def report(model=None):
    pass # report test result here later on

# run the following lines to stop mlflow server
# mlflow_process.terminate()  # Gracefully stop the server
# mlflow_process.wait()       # Wait for the process to finish
# print("MLflow server stopped.")
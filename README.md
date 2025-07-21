
# Industrializing a ML pipeline with MLOps

## 📝 Description

This end-to-end project serves as a blueprint for deploying ML models following **MLOps** principles. The **primary goal** is to demonstrate the journey from a trained model to a production-ready, industrialized pipeline. 

For this demonstration, I use a wine quality prediction model, specifically an Elastic Net regression, to predict wine quality based on its chemical composition (e.g., acidity, sugar, pH). (More info on this below)

The project spans the entire ML lifecycle from data extraction to model deployment, and monitoring and covers major aspects of MLOps, including :
* Pipeline automation and orchestration via Prefect
* Experiment tracking and model versioning via Mlflow
* Automated deployment and monitoring.
* Reproducible deployments and seamless scaling via Docker and Kubernetes
* Real-time visualization and alerting of drift metrics via Deepchecks and Grafana

## 🚀 Demo video 
Watch this brief demo video to see the complete process, from data preparation all the way through to model deployment and monitoring.

https://github.com/user-attachments/assets/c9bf8485-51b6-4fe6-8346-04702c296454


## 📦 Key Tools
* **🐳☸️ Docker & Kubernetes :** The entire project is containerized using Docker, and Kubernetes is used for multi-container deployments and orchestration. This includes services for _MLflow_, _Prefect_, _MinIO_, _Postgres_, and _Grafana_.  

* **📊 MLflow**: Used to track training experiments for easy comparison and model selection, and also to help version and manage models to streamline deployment. 

* **📦 MinIO**: Used as a S3-compatible object storage solution to save the pipeline's train data and MLflow artifacts.

* **⛓️ Prefect**: Used to orchestrate the project pipelines.

* **🎯 Hyperopt**: Used for hyperparameters tuning to explore exclusively promising regions.  

* **🚀 FastAPI**: Used to serve trained models and handle prediction requests in real-time.

* **🐘 PostgreSQL**: Hosts the backend databases for MLflow, Prefect, and monitoring.

* **🖥️ Adminer**: Adminer provided a light-weight front-end to manage and monitor the PostgreSQL database.

* **🧪 Deepchecks**: Used to calculate both Features and Prediction drift  

* **📈 Grafana**: Used to visualize drift scores and to provide alerts for drift detection, enabling real-time insights into deployed model health.  

* **⚙️ OmegaConf**: Used to manage configurations through structured YAML with runtime parameter injection

* **📦 Poetry**: Used to manage Python dependencies and virtual environments, for reproducible and consistent development.  

## 📊  Dataset

This project uses the **Red Wine Quality dataset from the UCI Machine Learning Repository** to predict the quality of red wine based on various chemical properties, such as alcohol content, acidity, and pH.

- [Download Red Wine Quality Dataset (UCI)](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)

## 🧙 Model 

The model is built using **ElasticNet** regression, a linear regression technique that balances Lasso (L1) and Ridge (L2) regularization.

**Hyperopt** is used for hyperparameter tuning to find the best values for `alpha` regularization strength and `l1_ratio` balance between L1 and L2 penalties.

## 📌 Prerequisites

- Make
- Docker
- Kubernetes cluster (minikube, kind, or managed)
- kubectl configured to access your cluster

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zakariajaadi/WinePredictionMlops.git
   cd WinePredictionMlops
   ```
2. Create dotenv file:

   Create a `.env` file at the root of the project and add the following environment variables:

   ```env
    ENV_MODE=prod

    # Postgres
    DB_USER=postgres
    DB_PASSWORD=example

    # Minio
    AWS_ACCESS_KEY_ID=minioadmin
    AWS_SECRET_ACCESS_KEY=minioadmin

    # Mlflow
    MLFLOW_S3_ENDPOINT_URL=http://minio:9000

    # Prefect
    #PREFECT_API_URL="http://localhost:30420/api"
    PREFECT_LOGGING_LEVEL=INFO

   
3. Build and Push flow image:
   ```bash
   make release TAG=1.0.0 # Builds and pushes a Docker image containing the application code and all required dependencies.
   ```
4. Deploy kubernetes resources:
   ```bash
   make deploy-k8s # Applies Kubernetes manifests
   ```
   Once all pods are running (check with `kubectl get pods`), you can access these services:
   
      * Prefect UI: `http://localhost:30420` 
      * MLflow UI: `http://localhost:30500` 
      * Minio UI: `http://localhost:32000`
      * Grafana: `http://localhost:30000` 
      * Adminer: `http://localhost:30081` 

5. Deploy flows in prefect:
   ```bash
   make deploy-all-flows # Deploys all Prefect flows to the Prefect server
   ```
6. Run the Training and Deployment flow in prefect UI:

   Access the prefect UI (`http://localhost:30420`), navigate to Deployments, to `wine_quality_ml_pipeline_production` and trigger a flow Run. 

## 🍽️ Model Serving

1. Serve model:
   ```bash
   make deploy-model-api # Exposes the champion model via a FastAPI application.
   ```
   
2. Check model API health:

   Visit the model health endpoint in your browser or with curl:

   ```bash
   curl http://localhost:30080/health
   ```
   You should receive a response like:
   ```json
   {"status": "ok"}
   ```
3. Test the model prediction with Postman

    1. Open Postman and create a `POST` request to: `http://localhost:30080/predict`
    2. In the request body, paste the sample payload below, and click Send.
        ```json
        {
          "inputs": [
            {
              "fixed acidity": 7.4,
              "volatile acidity": 0.7,
              "citric acid": 0,
              "residual sugar": 1.9,
              "chlorides": 0.076,
              "free sulfur dioxide": 11,
              "total sulfur dioxide": 34,
              "density": 0.9978,
              "pH": 3.51,
              "sulphates": 0.56,
              "alcohol": 9.4
            }
          ]
        }
       
    3. You should receive a JSON response like:
        ```json
               {
                "predictions": [5.1]
               }
        ```

## 📜 License
This project is licensed under the MIT License.

## 🚧 Future Improvements
- Integrate CI/CD.

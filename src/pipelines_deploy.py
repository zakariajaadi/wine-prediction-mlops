from dotenv import dotenv_values
from pipelines import ml_workflow, monitoring_workflow, inference_simulation_workflow

# Deploy all prefect pipelines
def deploy_all_flows():

    # ----- Deploy & Schedule Flows ---- #

    # Get env dict from .env
    env_vars = dotenv_values()
    env_vars["PREFECT_API_URL"] = "http://prefect:4200/api"

    # Deploy ml pipeline flow
    ml_workflow.deploy(
        name="wine_quality_ml_pipeline_production",
        cron="0 0 * * 0",  # every sunday midnight
        work_pool_name="my_k8s_pool",
        image="zakariajaadi/k8s-getting-started:1.0.0",
        job_variables={
            "image_pull_policy": "Always",
            "env": env_vars
        },
        build=False,
        push=False
    )

    # Deploy monitoring  flow
    monitoring_workflow.deploy(
        name="wine_quality_monitoring_production",
        work_pool_name="my_k8s_pool",
        image="zakariajaadi/k8s-getting-started:1.0.0",
        job_variables={
            "image_pull_policy": "Always",
            "env": env_vars
        },
        cron="0 12 * * *",  # Every day 12pm
        build=False,
        push=False
    )

    # Deploy monitoring  flow
    inference_simulation_workflow.deploy(
        name="wine_quality_inference_simulation",
        work_pool_name="my_k8s_pool",
        image="zakariajaadi/k8s-getting-started:1.0.0",
        job_variables={
            "image_pull_policy": "Always",
            "env": env_vars
        },
        build=False,
        push=False
    )

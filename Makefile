#buid flow image command
release:
	docker build -t zakariajaadi/k8s-getting-started:$(TAG) .
	docker push zakariajaadi/k8s-getting-started:$(TAG)


#k8s commands
deploy-app-config:
	kubectl apply -f k8s/app
deploy-postgres:
	kubectl apply -f k8s/postgres
deploy-mlflow:
	kubectl apply -f k8s/mlflow
deploy-prefect:
	kubectl apply -f k8s/prefect
deploy-adminer:
	kubectl apply -f k8s/adminer
deploy-grafana:
	kubectl apply -f k8s/grafana
deploy-minio:
	kubectl apply -f k8s/minio

deploy-k8s:
	kubectl apply -f k8s/app
	kubectl apply -f k8s/postgres
	kubectl apply -f k8s/mlflow
	kubectl apply -f k8s/prefect
	kubectl apply -f k8s/minio
	kubectl apply -f k8s/grafana
	kubectl apply -f k8s/adminer

model-serving:
	kubectl apply -f k8s/model


# Deploy all pipelines in prefect server ( important! : app profile docker compose must be up)
deploy-all-flows:
	python src/main.py





.PHONY: deploy-app-config deploy-postgres deploy-mlflow deploy-prefect deploy-adminer deploy-grafana deploy-minio deploy-model-api deploy-k8s





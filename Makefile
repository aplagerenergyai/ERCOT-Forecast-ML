# ============================================================================
# ERCOT ML Pipeline - Makefile
# 
# Quick commands for Docker build, push, run, and test
# ============================================================================

# Configuration
IMAGE_NAME := ercot-ml-pipeline
IMAGE_TAG := latest
REGISTRY := myregistry.azurecr.io
FULL_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# Model configuration
MODEL_TYPE := lgbm
PORT := 5001

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help build push run test shell clean stop logs train validate

help:  ## Show this help message
	@echo "$(GREEN)ERCOT ML Pipeline - Docker Commands$(NC)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Environment Variables:$(NC)"
	@echo "  IMAGE_NAME  = $(IMAGE_NAME)"
	@echo "  IMAGE_TAG   = $(IMAGE_TAG)"
	@echo "  REGISTRY    = $(REGISTRY)"
	@echo "  MODEL_TYPE  = $(MODEL_TYPE)"
	@echo "  PORT        = $(PORT)"

build:  ## Build Docker image
	@echo "$(GREEN)Building Docker image: $(IMAGE_NAME):$(IMAGE_TAG)$(NC)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "$(GREEN)✓ Build complete$(NC)"

build-no-cache:  ## Build Docker image without cache
	@echo "$(GREEN)Building Docker image (no cache): $(IMAGE_NAME):$(IMAGE_TAG)$(NC)"
	docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "$(GREEN)✓ Build complete$(NC)"

tag:  ## Tag image for registry
	@echo "$(GREEN)Tagging image: $(FULL_IMAGE)$(NC)"
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE)
	@echo "$(GREEN)✓ Tag complete$(NC)"

push: tag  ## Push image to container registry
	@echo "$(GREEN)Pushing image: $(FULL_IMAGE)$(NC)"
	docker push $(FULL_IMAGE)
	@echo "$(GREEN)✓ Push complete$(NC)"

run:  ## Run container locally (inference mode)
	@echo "$(GREEN)Starting container (Model: $(MODEL_TYPE), Port: $(PORT))$(NC)"
	docker run -d \
		--name $(IMAGE_NAME) \
		-p $(PORT):5001 \
		-e MODEL_TYPE=$(MODEL_TYPE) \
		-e LOG_LEVEL=INFO \
		-v $(PWD)/models:/app/models:ro \
		$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "$(GREEN)✓ Container started$(NC)"
	@echo "$(YELLOW)Test endpoint: http://localhost:$(PORT)/health$(NC)"

run-interactive:  ## Run container interactively
	@echo "$(GREEN)Starting container (interactive)$(NC)"
	docker run -it --rm \
		--name $(IMAGE_NAME)-interactive \
		-p $(PORT):5001 \
		-e MODEL_TYPE=$(MODEL_TYPE) \
		-e LOG_LEVEL=INFO \
		-v $(PWD)/models:/app/models:ro \
		-v $(PWD)/data:/app/data:ro \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		/bin/bash

train-lgbm:  ## Run LightGBM training in container
	@echo "$(GREEN)Running LightGBM training$(NC)"
	docker run --rm \
		--name $(IMAGE_NAME)-train-lgbm \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/models:/app/models \
		-e AZUREML_OUTPUT_model=/app/models/lgbm \
		-e AZUREML_INPUT_features=/app/data/features \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python train_lgbm.py
	@echo "$(GREEN)✓ Training complete$(NC)"

train-xgb:  ## Run XGBoost training in container
	@echo "$(GREEN)Running XGBoost training$(NC)"
	docker run --rm \
		--name $(IMAGE_NAME)-train-xgb \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/models:/app/models \
		-e AZUREML_OUTPUT_model=/app/models/xgb \
		-e AZUREML_INPUT_features=/app/data/features \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python train_xgb.py
	@echo "$(GREEN)✓ Training complete$(NC)"

train-deep:  ## Run Deep Learning training in container
	@echo "$(GREEN)Running Deep Learning training$(NC)"
	docker run --rm \
		--name $(IMAGE_NAME)-train-deep \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/models:/app/models \
		-e AZUREML_OUTPUT_model=/app/models/deep \
		-e AZUREML_INPUT_features=/app/data/features \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		python train_deep.py
	@echo "$(GREEN)✓ Training complete$(NC)"

test:  ## Test local scoring endpoint
	@echo "$(GREEN)Testing scoring endpoint$(NC)"
	@bash test_score_local.sh

test-health:  ## Test health endpoint
	@echo "$(GREEN)Testing health endpoint$(NC)"
	@curl -s http://localhost:$(PORT)/health | python -m json.tool

test-info:  ## Get model info
	@echo "$(GREEN)Getting model info$(NC)"
	@curl -s http://localhost:$(PORT)/model/info | python -m json.tool

shell:  ## Open bash shell inside running container
	@echo "$(GREEN)Opening shell in container$(NC)"
	docker exec -it $(IMAGE_NAME) /bin/bash

logs:  ## Show container logs
	@echo "$(GREEN)Container logs:$(NC)"
	docker logs -f $(IMAGE_NAME)

stop:  ## Stop running container
	@echo "$(YELLOW)Stopping container$(NC)"
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true
	@echo "$(GREEN)✓ Container stopped$(NC)"

clean: stop  ## Clean up containers and images
	@echo "$(YELLOW)Cleaning up Docker artifacts$(NC)"
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	docker system prune -f
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

ps:  ## Show running containers
	@echo "$(GREEN)Running containers:$(NC)"
	@docker ps --filter "name=$(IMAGE_NAME)"

images:  ## List built images
	@echo "$(GREEN)Built images:$(NC)"
	@docker images | grep $(IMAGE_NAME) || echo "No images found"

# ============================================================================
# Azure ML Training Jobs
# ============================================================================

validate:  ## Validate parquet features file
	@echo "$(GREEN)Validating feature parquet file$(NC)"
	@python validate_parquet.py

train:  ## Submit all training jobs to Azure ML
	@echo "$(GREEN)Submitting training jobs to Azure ML$(NC)"
	@bash run_training_jobs.sh

# Azure Container Registry login
acr-login:  ## Login to Azure Container Registry
	@echo "$(GREEN)Logging in to ACR: $(REGISTRY)$(NC)"
	az acr login --name $(shell echo $(REGISTRY) | cut -d. -f1)
	@echo "$(GREEN)✓ Login successful$(NC)"

# Complete workflow
all: build run test  ## Build, run, and test

# Production workflow
deploy: build tag acr-login push  ## Build, tag, and push to ACR


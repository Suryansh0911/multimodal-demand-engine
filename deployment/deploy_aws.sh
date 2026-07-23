#!/bin/bash
set -e

# --- Configuration ---
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="950693198832"
REPO_NAME="demand-engine"
IMAGE_TAG="latest"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "=== 1. Authenticating with AWS ECR ==="
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "=== 2. Creating ECR Repository (if it doesn't exist) ==="
aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

echo "=== 3. Building Production Docker Image ==="
docker build -t $REPO_NAME -f docker/Dockerfile.serving .

echo "=== 4. Tagging and Pushing Image to ECR ==="
docker tag ${REPO_NAME}:${IMAGE_TAG} $ECR_URI
docker push $ECR_URI

echo "=== 5. Deploying to AWS App Runner ==="
aws apprunner create-service --cli-input-json file://deployment/apprunner.json

echo "Deployment initiated! Check the AWS App Runner console for the live URL."
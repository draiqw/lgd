#!/bin/bash

set -e  # Exit on error

DOCKER_USERNAME="${DOCKER_USERNAME:-YOUR_DOCKERHUB_USERNAME}"
IMAGE_NAME="llabs_lda_hyperopt"
VERSION="${1:-v1}"

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "============================================================================"
echo "Docker Build and Push Script"
echo "============================================================================"
print_info "Image: ${FULL_IMAGE_NAME}"
echo "============================================================================"
echo ""


if [[ "$OSTYPE" == "darwin"* ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        print_info "Detected Apple Silicon Mac (ARM64)"
        PLATFORM_FLAG="--platform linux/amd64"
        BUILD_CMD="docker buildx build"
    else
        print_info "Detected Intel Mac (x86_64)"
        PLATFORM_FLAG=""
        BUILD_CMD="docker build"
    fi
else
    print_info "Detected Linux"
    PLATFORM_FLAG=""
    BUILD_CMD="docker build"
fi

# Step 1: Build the image
print_step "Building Docker image..."
if [ -n "$PLATFORM_FLAG" ]; then
    $BUILD_CMD $PLATFORM_FLAG -t "$IMAGE_NAME" .
else
    $BUILD_CMD -t "$IMAGE_NAME" .
fi

if [ $? -eq 0 ]; then
    print_info "✓ Build successful"
else
    print_error "Build failed!"
    exit 1
fi

echo ""

# Step 2: Tag the image
print_step "Tagging image as ${FULL_IMAGE_NAME}..."
docker tag "$IMAGE_NAME" "$FULL_IMAGE_NAME"

if [ $? -eq 0 ]; then
    print_info "✓ Tag successful"
else
    print_error "Tagging failed!"
    exit 1
fi

echo ""

# Step 3: Push to Docker Hub
print_step "Pushing to Docker Hub..."
print_info "You may need to login first: docker login"

docker push "$FULL_IMAGE_NAME"

if [ $? -eq 0 ]; then
    print_info "✓ Push successful"
else
    print_error "Push failed! Make sure you're logged in: docker login"
    exit 1
fi

echo ""
echo "============================================================================"
echo -e "${GREEN}SUCCESS!${NC} Image pushed to Docker Hub"
echo "============================================================================"
echo ""
echo "Image name: ${FULL_IMAGE_NAME}"
echo ""
echo "Next steps:"
echo "  1. Convert to .sqsh on enroot VM:"
echo "     ssh -p 2295 mmp@188.44.41.125"
echo "     sudo enroot import docker://${FULL_IMAGE_NAME}"
echo ""
echo "  2. Copy .sqsh to slurm-master (see run_exp.md for details)"
echo ""
echo "============================================================================"
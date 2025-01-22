#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; docker system prune -f; }
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
docker build -t cpu-test -f Dockerfile.ppc64le .

# Run the image, setting --shm-size=4g for tensor parallel.
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true --network host -e HF_TOKEN --name cpu-test cpu-test

function cpu_tests() {
  set -e
  
  # offline inference
  docker exec cpu-test bash -c "
    set -e
    export PATH=/opt/conda/bin:$PATH
    python3 examples/offline_inference/basic.py"

  # Run basic model test
  docker exec cpu-test bash -c "
    set -e
    pytest -v -s tests/models/embedding/language/test_cls_models.py::test_classification_models[float-jason9693/Qwen2.5-1.5B-apeach]
    pytest -v -s tests/models/embedding/language/test_embedding.py::test_models[half-BAAI/bge-base-en-v1.5]
    pytest -v -s tests/models/encoder_decoder/language -m cpu_model
    pytest -v -s tests/models/decoder_only/audio_language/test_ultravox.py::test_online_inference[server0]
    pytest -v -s tests/models/decoder_only/vision_language/test_models.py::test_video_models[qwen2_vl-test_case0]
    pytest -v -s tests/models/decoder_only/vision_language/test_models.py::test_image_embedding_models[llava-test_case0]
    pytest -v -s tests/models/decoder_only/vision_language/test_models.py::test_multi_image_models[qwen2_vl-test_case20]
    pytest -v -s tests/models/decoder_only/vision_language/test_models.py::test_single_image_models[qwen2_vl-test_case20]
    pytest -v -s tests/models/decoder_only/vision_language/test_models.py::test_single_image_models[llava-test_case56]
    pytest -v -s tests/models/decoder_only/vision_language -m cpu_model"
}

# All of CPU tests are expected to be finished less than 40 mins.
export -f cpu_tests
timeout 40m bash -c cpu_tests

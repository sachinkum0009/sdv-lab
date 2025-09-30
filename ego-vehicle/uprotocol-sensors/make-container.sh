#!/usr/bin/env bash
# Works whether you're using sudo or not
USER_UID=$(id -u "${SUDO_USER:-$USER}")
USER_GID=$(id -g "${SUDO_USER:-$USER}")

TARGET=${1:-DEV}

sudo podman build \
  --target $TARGET \
  --build-arg USER_UID=$USER_UID \
  --build-arg USER_GID=$USER_GID \
  --build-arg USERNAME=dev \
  -t uprotocol-sensors .
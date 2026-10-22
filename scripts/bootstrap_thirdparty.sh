#!/usr/bin/env bash
# Download ONNX Runtime prebuilt bundles into wbc_g1_deploy/thirdparty/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_VERSION="${ORT_VERSION:-1.30.0}"
ORT_BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"
THIRDPARTY="${ROOT}/thirdparty"

mkdir -p "${THIRDPARTY}"

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${out}" "${url}"
  else
    echo "Need curl or wget to download ONNX Runtime." >&2
    exit 1
  fi
}

install_ort() {
  local name="$1"
  local dest="${THIRDPARTY}/${name}"

  if [[ -f "${dest}/include/onnxruntime_c_api.h" ]]; then
    echo "Already installed: ${name}"
  else
    local url="${ORT_BASE_URL}/${name}.tgz"
    local tmp
    tmp="$(mktemp -d)"

    echo "Downloading ${name}..."
    download "${url}" "${tmp}/${name}.tgz"
    rm -rf "${dest}"
    tar -xzf "${tmp}/${name}.tgz" -C "${THIRDPARTY}"
    rm -rf "${tmp}"
    echo "Installed ${name}"
  fi

  local targets="${dest}/lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake"
  if [[ -f "${targets}" ]] \
    && grep -q 'lib64/libonnxruntime' "${targets}" \
    && [[ ! -e "${dest}/lib64" ]] \
    && [[ -d "${dest}/lib" ]]; then
    ln -s lib "${dest}/lib64"
    echo "Added lib64 compatibility link for ${name}"
  fi
}

for ort in onnxruntime-linux-x64-${ORT_VERSION} onnxruntime-linux-aarch64-${ORT_VERSION}; do
  install_ort "${ort}"
done

echo "Done. thirdparty at ${THIRDPARTY}"

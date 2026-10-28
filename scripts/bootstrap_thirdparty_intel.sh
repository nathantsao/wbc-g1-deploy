#!/usr/bin/env bash
# Build the ONNX Runtime OpenVINO execution provider under thirdparty/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVEP_VERSION="${OVEP_VERSION:-1.8.0}"
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.30.0}"
OPENVINO_REF="${OPENVINO_REF:-master}"
OVEP_BASE_URL="https://github.com/onnxruntime/onnxruntime-ep-openvino/archive/refs/tags/"
OPENVINO_REPO="https://github.com/openvinotoolkit/openvino.git"
THIRDPARTY="${ROOT}/thirdparty"
OPENVINO_SOURCE="${THIRDPARTY}/openvino"
OPENVINO_BUILD="${THIRDPARTY}/build/openvino"
OPENVINO_INSTALL="${THIRDPARTY}/openvino-install"

mkdir -p "${THIRDPARTY}"

download_source() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${out}" "${url}"
  else
    echo "Need curl or wget to download ONNX Runtime OpenVINO EP source." >&2
    exit 1
  fi
}

install_dependencies() {
  local ort_arch ort_name ep_source tmp

  if [[ -z "${ONNXRUNTIME_ROOT_DIR:-}" ]]; then
    case "$(uname -m)" in
      x86_64) ort_arch="x64" ;;
      aarch64) ort_arch="aarch64" ;;
      *)
        echo "Unsupported architecture for the ONNX Runtime package: $(uname -m)" >&2
        return 1
        ;;
    esac

    ORT_VERSION="${ONNXRUNTIME_VERSION}" "${ROOT}/scripts/bootstrap_thirdparty.sh"
    ort_name="onnxruntime-linux-${ort_arch}-${ONNXRUNTIME_VERSION}"
    ONNXRUNTIME_ROOT_DIR="${THIRDPARTY}/${ort_name}"
  fi

  if [[ ! -f "${ONNXRUNTIME_ROOT_DIR}/lib/cmake/onnxruntime/onnxruntimeConfig.cmake" ]]; then
    echo "ONNX Runtime package not found at ${ONNXRUNTIME_ROOT_DIR}." >&2
    return 1
  fi
  export ONNXRUNTIME_ROOT_DIR
  echo "Using ONNXRUNTIME_ROOT_DIR=${ONNXRUNTIME_ROOT_DIR}"

  ep_source="${THIRDPARTY}/onnxruntime-ep-openvino-${OVEP_VERSION}"
  if [[ ! -f "${ep_source}/CMakeLists.txt" ]]; then
    tmp="$(mktemp -d)"
    echo "Downloading ONNX Runtime OpenVINO EP ${OVEP_VERSION}..."
    if ! download_source "${OVEP_BASE_URL}/v${OVEP_VERSION}.tar.gz" "${tmp}/source.tar.gz"; then
      rm -rf "${tmp}"
      return 1
    fi
    mkdir -p "${ep_source}"
    if ! tar -xzf "${tmp}/source.tar.gz" --strip-components=1 -C "${ep_source}"; then
      rm -rf "${tmp}"
      return 1
    fi
    rm -rf "${tmp}"
  else
    echo "Using existing source: ${ep_source}"
  fi

  if [[ ! -d "${OPENVINO_SOURCE}/.git" ]]; then
    if [[ -e "${OPENVINO_SOURCE}" ]]; then
      echo "OpenVINO source path exists but is not a git checkout: ${OPENVINO_SOURCE}" >&2
      return 1
    fi
    echo "Cloning OpenVINO ${OPENVINO_REF} with submodules..."
    git clone --branch "${OPENVINO_REF}" --recursive "${OPENVINO_REPO}" "${OPENVINO_SOURCE}"
  else
    echo "Using existing OpenVINO source: ${OPENVINO_SOURCE}"
    git -C "${OPENVINO_SOURCE}" submodule update --init --recursive
  fi

  if [[ ! -f "${OPENVINO_BUILD}/CMakeCache.txt" ]]; then
    echo "Installing OpenVINO build dependencies (sudo required)..."
    sudo "${OPENVINO_SOURCE}/install_build_dependencies.sh"
  fi

  cmake \
    -S "${OPENVINO_SOURCE}" \
    -B "${OPENVINO_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OPENVINO_INSTALL}"
  cmake --build "${OPENVINO_BUILD}" --parallel -j 4
  cmake --install "${OPENVINO_BUILD}"

  if [[ ! -f "${OPENVINO_INSTALL}/setupvars.sh" ]]; then
    echo "OpenVINO setup script not found at ${OPENVINO_INSTALL}/setupvars.sh." >&2
    return 1
  fi
  # The EP build uses these variables to locate OpenVINO's CMake package and runtime libraries.
  source "${OPENVINO_INSTALL}/setupvars.sh"
  echo "Using OpenVINO from ${INTEL_OPENVINO_DIR}"
}

install_openvino_ep() {
  local source="${THIRDPARTY}/onnxruntime-ep-openvino-${OVEP_VERSION}"
  local build_dir="${THIRDPARTY}/build/onnxruntime-ep-openvino-${OVEP_VERSION}"
  local install_dir="${THIRDPARTY}/onnxruntime-ep-openvino-${OVEP_VERSION}-install"

  cmake \
    -S "${source}" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_dir}" \
    -DOVEP_PLUGIN_BUILD_TESTS=OFF \
    -DOVEP_PLUGIN_BUILD_TOOLS=OFF
  cmake --build "${build_dir}" --target install --parallel -j 4

  echo "Installed OpenVINO EP to ${install_dir}"
}

install_dependencies
install_openvino_ep

echo "Done. thirdparty at ${THIRDPARTY}"

#!/usr/bin/env bash
# Install OpenVINO C++ runtime and build the ONNX Runtime OpenVINO EP plugin under thirdparty/.
#
# Usage:
#   scripts/bootstrap_thirdparty_intel.sh                    # ORT + OpenVINO + OpenVINO EP
#   scripts/bootstrap_thirdparty_intel.sh --system-deps      # also install build tools and GPU/NPU drivers (sudo)
#   scripts/bootstrap_thirdparty_intel.sh --system-deps-only # only install build tools and GPU/NPU drivers (sudo)
#   scripts/bootstrap_thirdparty_intel.sh --skip-ep          # skip building the OpenVINO EP plugin
#
# Intel publishes the NPU driver only for some Ubuntu releases. On others, set
# NPU_DRIVER_DISTRO to a published one, e.g. NPU_DRIVER_DISTRO=ubuntu2404.
# --system-deps keeps an installed NPU driver. To install a specific version instead,
# set NPU_DRIVER_TAG and NPU_DRIVER_BUILD.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRDPARTY="${ROOT}/thirdparty"

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.30.0}"
OPENVINO_VERSION="${OPENVINO_VERSION:-2026.4.1}"
OPENVINO_BUILD="${OPENVINO_BUILD:-22982.07f9c262b05}"
OVEP_VERSION="${OVEP_VERSION:-1.8.1}"
# Remember whether the user set NPU_DRIVER_TAG. If not, an installed NPU driver is kept.
NPU_DRIVER_REQUESTED="${NPU_DRIVER_TAG:-}"
NPU_DRIVER_TAG="${NPU_DRIVER_TAG:-v1.38.0}"
NPU_DRIVER_BUILD="${NPU_DRIVER_BUILD:-20260910-34487311128}"
JOBS="${JOBS:-$(nproc)}"

OPENVINO_DIR="${THIRDPARTY}/openvino-${OPENVINO_VERSION}"
OVEP_SOURCE="${THIRDPARTY}/onnxruntime-ep-openvino-${OVEP_VERSION}"
OVEP_BUILD="${THIRDPARTY}/build/onnxruntime-ep-openvino-${OVEP_VERSION}"
OVEP_INSTALL="${THIRDPARTY}/onnxruntime-ep-openvino-${OVEP_VERSION}-install"

WITH_SYSTEM_DEPS=0
SYSTEM_DEPS_ONLY=0
SKIP_EP=0
for arg in "$@"; do
  case "${arg}" in
    --system-deps) WITH_SYSTEM_DEPS=1 ;;
    --system-deps-only) WITH_SYSTEM_DEPS=1; SYSTEM_DEPS_ONLY=1 ;;
    --skip-ep) SKIP_EP=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
  esac
done

download() {
  local url="$1"
  local out="$2"
  echo "Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL# --retry 3 -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${out}" "${url}"
  else
    echo "Need curl or wget." >&2
    exit 1
  fi
}

ubuntu_version() {
  # e.g. "26.04"
  . /etc/os-release
  if [[ "${ID}" != "ubuntu" ]]; then
    echo "Only Ubuntu is supported (found ${ID})." >&2
    return 1
  fi
  echo "${VERSION_ID}"
}

ubuntu_major() {
  # e.g. "26.04" -> "26"
  local version
  version="$(ubuntu_version)"
  echo "${version%%.*}"
}

install_system_deps() {
  local version npu_distro tmp user installed
  version="$(ubuntu_version)"
  # Intel publishes NPU packages only for some releases (e.g. ubuntu2404, ubuntu2604).
  npu_distro="${NPU_DRIVER_DISTRO:-ubuntu${version//./}}"
  echo "== Installing build tools and Intel GPU/NPU user-mode drivers (sudo required)"

  # Intel's recommended GPU driver and Level Zero loader are newer than the Ubuntu archive.
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake ninja-build git curl ca-certificates \
    libtbb12 libze1 libze-intel-gpu1 intel-opencl-icd clinfo

  # Device ACLs only cover local desktop sessions; group membership also covers SSH.
  user="${SUDO_USER:-$(id -un)}"
  if ! id -nG "${user}" | grep -qw render; then
    sudo gpasswd -a "${user}" render
    echo "Added ${user} to the render group; log out and back in (or reboot) to apply."
  fi

  # NPU driver is only published on GitHub, not in apt.
  # Keep an installed NPU driver unless NPU_DRIVER_TAG is set.
  # shellcheck disable=SC2016  # ${db:Status-Status} and ${Version} are dpkg-query fields
  installed="$(dpkg-query -W -f='${db:Status-Status} ${Version}' intel-level-zero-npu 2>/dev/null || true)"
  if [[ "${installed}" == "installed "* && -z "${NPU_DRIVER_REQUESTED}" ]]; then
    echo "NPU driver ${installed##* } is already installed and will be kept. Set NPU_DRIVER_TAG to install a specific version."
    return 0
  fi
  tmp="$(mktemp -d)"
  if ! download \
    "https://github.com/intel/linux-npu-driver/releases/download/${NPU_DRIVER_TAG}/linux-npu-driver-${NPU_DRIVER_TAG}.${NPU_DRIVER_BUILD}-${npu_distro}.tar.gz" \
    "${tmp}/npu.tar.gz"; then
    echo "No NPU driver ${NPU_DRIVER_TAG} package for ${npu_distro}." >&2
    echo "See https://github.com/intel/linux-npu-driver/releases/tag/${NPU_DRIVER_TAG} and set NPU_DRIVER_DISTRO (e.g. ubuntu2404)." >&2
    rm -rf "${tmp}"
    return 1
  fi
  tar -xzf "${tmp}/npu.tar.gz" -C "${tmp}"
  sudo dpkg --purge --force-remove-reinstreq \
    intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym || true
  sudo dpkg -i "${tmp}"/*.deb
  rm -rf "${tmp}"
}

install_onnxruntime() {
  local ort_arch
  case "$(uname -m)" in
    x86_64) ort_arch="x64" ;;
    aarch64) ort_arch="aarch64" ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; return 1 ;;
  esac

  if [[ -z "${ONNXRUNTIME_ROOT_DIR:-}" ]]; then
    ORT_VERSION="${ONNXRUNTIME_VERSION}" "${ROOT}/scripts/bootstrap_thirdparty.sh"
    ONNXRUNTIME_ROOT_DIR="${THIRDPARTY}/onnxruntime-linux-${ort_arch}-${ONNXRUNTIME_VERSION}"
  fi
  if [[ ! -f "${ONNXRUNTIME_ROOT_DIR}/lib/cmake/onnxruntime/onnxruntimeConfig.cmake" ]]; then
    echo "ONNX Runtime package not found at ${ONNXRUNTIME_ROOT_DIR}." >&2
    return 1
  fi
  # The release tarball keeps headers in include/, but its CMake config and the EP expect include/onnxruntime/.
  if [[ -f "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime_c_api.h" ]] \
    && [[ ! -e "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime" ]]; then
    ln -s . "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime"
    echo "Added include/onnxruntime compatibility link"
  fi
  export ONNXRUNTIME_ROOT_DIR
  echo "Using ONNXRUNTIME_ROOT_DIR=${ONNXRUNTIME_ROOT_DIR}"
}

install_openvino() {
  local major name tmp
  if [[ -f "${OPENVINO_DIR}/setupvars.sh" ]]; then
    echo "Already installed: OpenVINO ${OPENVINO_VERSION}"
    return
  fi
  if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This script only fetches the x86_64 OpenVINO archive." >&2
    return 1
  fi

  # Prebuilt archive is used instead of building OpenVINO from source (hours, unpinned).
  major="${OPENVINO_DISTRO:-ubuntu$(ubuntu_major)}"
  name="openvino_toolkit_${major}_${OPENVINO_VERSION}.${OPENVINO_BUILD}_x86_64"
  tmp="$(mktemp -d)"
  download \
    "https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION}/linux/${name}.tgz" \
    "${tmp}/openvino.tgz"
  rm -rf "${OPENVINO_DIR}"
  mkdir -p "${OPENVINO_DIR}"
  tar -xzf "${tmp}/openvino.tgz" --strip-components=1 -C "${OPENVINO_DIR}"
  rm -rf "${tmp}"
  echo "Installed OpenVINO ${OPENVINO_VERSION} to ${OPENVINO_DIR}"
}

source_openvino() {
  # setupvars.sh reads unset variables, which conflicts with set -u.
  set +u
  # shellcheck disable=SC1091
  source "${OPENVINO_DIR}/setupvars.sh" >/dev/null
  set -u
  echo "Using OpenVINO from ${INTEL_OPENVINO_DIR}"
}

install_openvino_ep() {
  local tmp generator=()
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required: the OpenVINO EP build fetches GSL/nlohmann_json with CMake FetchContent." >&2
    return 1
  fi

  if [[ ! -f "${OVEP_SOURCE}/CMakeLists.txt" ]]; then
    tmp="$(mktemp -d)"
    download \
      "https://github.com/onnxruntime/onnxruntime-ep-openvino/archive/refs/tags/v${OVEP_VERSION}.tar.gz" \
      "${tmp}/source.tar.gz"
    mkdir -p "${OVEP_SOURCE}"
    tar -xzf "${tmp}/source.tar.gz" --strip-components=1 -C "${OVEP_SOURCE}"
    rm -rf "${tmp}"
  else
    echo "Using existing source: ${OVEP_SOURCE}"
  fi

  # Choose a generator only for a new build directory. CMake cannot change it later.
  if [[ ! -f "${OVEP_BUILD}/CMakeCache.txt" ]] && command -v ninja >/dev/null 2>&1; then
    generator=(-G Ninja)
  fi
  # GIT_COMMIT_ID is set explicitly because the release tarball has no .git directory.
  cmake \
    -S "${OVEP_SOURCE}" \
    -B "${OVEP_BUILD}" \
    "${generator[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OVEP_INSTALL}" \
    -DBUILD_SHARED_LIBS=ON \
    -DGIT_COMMIT_ID="v${OVEP_VERSION}" \
    -DOVEP_PLUGIN_BUILD_TESTS=OFF \
    -DOVEP_PLUGIN_BUILD_TOOLS=OFF
  cmake --build "${OVEP_BUILD}" --target install --parallel "${JOBS}"

  OVEP_LIBRARY="$(find "${OVEP_INSTALL}" -name 'libonnxruntime_providers_openvino_plugin.so' | head -n1)"
  if [[ -z "${OVEP_LIBRARY}" ]]; then
    echo "OpenVINO EP plugin library not found under ${OVEP_INSTALL}." >&2
    return 1
  fi
  echo "Installed OpenVINO EP plugin: ${OVEP_LIBRARY}"
}

mkdir -p "${THIRDPARTY}"

if [[ "${WITH_SYSTEM_DEPS}" == 1 ]]; then
  install_system_deps
  [[ "${SYSTEM_DEPS_ONLY}" == 1 ]] && { echo "Done. System dependencies installed."; exit 0; }
fi

install_onnxruntime
install_openvino
source_openvino
[[ "${SKIP_EP}" == 1 ]] || install_openvino_ep

echo
echo "Done. thirdparty at ${THIRDPARTY}"
echo "Configure the benchmark with:"
echo "  cmake -S benchmark -B build/benchmark -DOpenVINO_DIR=${OPENVINO_DIR}/runtime/cmake"
if [[ -n "${OVEP_LIBRARY:-}" ]]; then
  echo "Use the OpenVINO EP with:"
  echo "  --runtime openvino-ep --provider-library ${OVEP_LIBRARY}"
fi

# Sourced by run_all.sh and compare_outputs.sh.

# build_default_bin ROOT TARGET rebuilds TARGET in build/benchmark, so the binary matches the
# source tree that print_provenance describes. Nothing is compiled if the build is up to date.
build_default_bin() {
  local build_dir="$1/build/benchmark"
  if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    echo "${build_dir} is not configured. See benchmark/README.md, or set BIN." >&2
    return 1
  fi
  # Build output goes to stderr, so it cannot mix with the CSV that compare_outputs.sh prints.
  echo "Building $2 in ${build_dir} so it matches the source tree." >&2
  cmake --build "${build_dir}" --target "$2" --parallel >&2
}

# print_provenance ROOT EP_LIB prints the code, plugin, and driver versions that a result depends
# on, as key=value lines for environment.txt.
print_provenance() {
  local root="$1" ep_lib="$2" dirty=unknown drivers
  echo "git_commit=$(git -C "${root}" rev-parse HEAD 2>/dev/null || echo unknown)"
  # With uncommitted changes to tracked files, git_commit alone does not identify the code.
  if git -C "${root}" rev-parse --git-dir > /dev/null 2>&1; then
    [[ -n "$(git -C "${root}" status --porcelain --untracked-files=no)" ]] && dirty=true || dirty=false
  fi
  echo "git_dirty=${dirty}"
  echo "ep_lib=${ep_lib:+$(realpath "${ep_lib}")}"
  # dpkg-query fails for packages that are not installed but still prints the installed ones.
  drivers="$(dpkg-query -W -f='${Package}=${Version} ' libze1 libze-intel-gpu1 intel-opencl-icd \
    intel-level-zero-npu intel-driver-compiler-npu 2>/dev/null || true)"
  drivers="${drivers% }"
  echo "drivers=${drivers:-unknown}"
}

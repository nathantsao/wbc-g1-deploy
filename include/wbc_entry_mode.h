#pragma once

#include <string>

namespace wbc_deploy {

enum class WbcEntryMode
{
  Standing,
  FromFloor,
  SelectedClip,
};

/// Set by FixStand, FloorReady, or WbcReady before entering Wbc_Tracking.
inline WbcEntryMode& pendingWbcEntryMode()
{
  static WbcEntryMode mode = WbcEntryMode::Standing;
  return mode;
}

inline std::string& pendingWbcClipName()
{
  static std::string name;
  return name;
}

}  // namespace wbc_deploy

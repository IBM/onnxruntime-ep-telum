// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <cstdint>

namespace telum_profile {

// Enable lightweight profiling for the Telum plugin EP with:
//   ORT_TELUM_PLUGIN_EP_PROFILE=1
//
// This is intended for local performance analysis only. When disabled, overhead should be near-zero.
inline constexpr const char* kTelumProfileEnvVar = "ORT_TELUM_PLUGIN_EP_PROFILE";

enum class Event : uint8_t {
  kCreateEpImpl = 0,
  kCreateEpImpl_ParseConfig,
  kReleaseEpImpl,
  kTelumEpCtor,
  kCreateTelumBackend,
  kGetCapabilityImpl,
  kCompileImpl,
  kSaveConstantInitializers,
  kCreateEpContextNodes,
  kNumEvents,
};

// Returns true if profiling is enabled (via env var). Cached on first use.
bool Enabled() noexcept;

// Adds timing for a given event. Thread-safe.
void Add(Event event, uint64_t duration_ns) noexcept;

// RAII scope timer. When profiling is disabled, it does nothing.
class ScopedEvent final {
 public:
  explicit ScopedEvent(Event event) noexcept
      : event_{event},
        enabled_{Enabled()},
        start_{enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}} {}

  ScopedEvent(const ScopedEvent&) = delete;
  ScopedEvent& operator=(const ScopedEvent&) = delete;

  ~ScopedEvent() noexcept {
    if (!enabled_) {
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    Add(event_, static_cast<uint64_t>(ns));
  }

 private:
  Event event_{};
  bool enabled_{false};
  std::chrono::steady_clock::time_point start_{};
};

}  // namespace telum_profile

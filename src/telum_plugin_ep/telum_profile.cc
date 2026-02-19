// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_profile.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace telum_profile {
namespace {

constexpr size_t kNumEvents = static_cast<size_t>(Event::kNumEvents);

const char* EventName(Event event) noexcept {
  switch (event) {
    case Event::kCreateEpImpl:
      return "CreateEpImpl";
    case Event::kCreateEpImpl_ParseConfig:
      return "CreateEpImpl.ParseConfig";
    case Event::kReleaseEpImpl:
      return "ReleaseEpImpl";
    case Event::kTelumEpCtor:
      return "TelumEp.ctor";
    case Event::kCreateTelumBackend:
      return "CreateTelumBackend";
    case Event::kGetCapabilityImpl:
      return "GetCapabilityImpl";
    case Event::kCompileImpl:
      return "CompileImpl";
    case Event::kSaveConstantInitializers:
      return "SaveConstantInitializers";
    case Event::kCreateEpContextNodes:
      return "CreateEpContextNodes";
    case Event::kNumEvents:
      break;
  }
  return "Unknown";
}

bool ParseEnabledEnv() noexcept {
  const char* v = std::getenv(kTelumProfileEnvVar);
  if (v == nullptr || *v == '\0') {
    return false;
  }
  return (std::strcmp(v, "1") == 0) || (std::strcmp(v, "true") == 0) || (std::strcmp(v, "TRUE") == 0);
}

struct ProfileState final {
  bool enabled{false};
  std::array<std::atomic<uint64_t>, kNumEvents> total_ns{};
  std::array<std::atomic<uint64_t>, kNumEvents> count{};

  ProfileState() noexcept : enabled(ParseEnabledEnv()) {
    for (size_t i = 0; i < kNumEvents; ++i) {
      total_ns[i].store(0, std::memory_order_relaxed);
      count[i].store(0, std::memory_order_relaxed);
    }
  }

  ~ProfileState() noexcept {
    if (!enabled) {
      return;
    }

    // Print a summary when the shared library is unloaded (tests unregister the EP library).
    // Emit to stderr to avoid interfering with PERF_RESULT parsing.
    std::fprintf(stderr, "TELUM_EP_PROFILE,env=%s\n", kTelumProfileEnvVar);
    for (size_t i = 0; i < kNumEvents; ++i) {
      const uint64_t ns = total_ns[i].load(std::memory_order_relaxed);
      const uint64_t c = count[i].load(std::memory_order_relaxed);
      if (c == 0) {
        continue;
      }
      const double total_us = static_cast<double>(ns) / 1000.0;
      const double avg_us = total_us / static_cast<double>(c);
      std::fprintf(stderr,
                   "TELUM_EP_PROFILE,event=%s,count=%" PRIu64 ",total_us=%.3f,avg_us=%.3f\n",
                   EventName(static_cast<Event>(i)), c, total_us, avg_us);
    }
    std::fprintf(stderr, "TELUM_EP_PROFILE,done=1\n");
  }
};

ProfileState& State() noexcept {
  static ProfileState state{};
  return state;
}

}  // namespace

bool Enabled() noexcept {
  return State().enabled;
}

void Add(Event event, uint64_t duration_ns) noexcept {
  ProfileState& s = State();
  if (!s.enabled) {
    return;
  }

  const size_t idx = static_cast<size_t>(event);
  if (idx >= kNumEvents) {
    return;
  }

  s.total_ns[idx].fetch_add(duration_ns, std::memory_order_relaxed);
  s.count[idx].fetch_add(1, std::memory_order_relaxed);
}

}  // namespace telum_profile

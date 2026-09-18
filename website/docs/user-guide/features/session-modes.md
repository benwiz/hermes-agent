---
title: Session modes and heavy tools
sidebar_position: 8
---

# Session modes and heavy tools

Session routing can expose `lean`, `coding`, and `heavy` tiers. The Discord
configuration uses `heavy` as the default for every channel; channel overrides
must not silently make one channel more privileged than another.

Heavy tools are available when useful, but the agent is instructed to use them
judiciously. Structured APIs, file/search/terminal tools, and public
search/extraction remain the preferred paths when they can complete the task.
This is guidance, not an availability gate. `/mode off` switches the current
conversation back to its configured default policy; `/mode heavy` restores the
heavy tier explicitly.

## Computer use

Set `computer_use.permission_mode: unrestricted` only when the operator wants
full desktop capability without repetitive per-action consent prompts. This
removes the computer-use approval loop but does not bypass authentication,
secret protections, or Hermes hard-blocked actions. Invalid permission modes
fail closed to `standard`.

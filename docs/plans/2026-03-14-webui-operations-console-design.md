# WebUI Operations Console Redesign

## Goal

Refocus the WebUI around remote gateway operations and status monitoring for a home-hosted Mac mini accessed over Tailscale. Chat remains available as a fallback path, but it should no longer dominate the information architecture.

## Problems

- Top-level navigation mixes unrelated mental models: chat, gateway control, workspace, agent tooling, and settings all compete equally.
- The overview page is overloaded with setup/configuration content instead of acting as a fast operational dashboard.
- Important operational details are spread across too many first-level tabs, creating duplicated entry points and weak hierarchy.
- Chat currently feels like the center of the product, even though the main usage pattern is remote operations and troubleshooting.

## Primary Usage Model

- Primary: remote operations console for gateway health, activity, channel status, sessions, cron, logs, and approvals.
- Secondary: drill into agent- and workspace-specific details when needed.
- Tertiary: open a direct chat session when other messaging paths fail.

## Information Architecture

Top-level sidebar tabs become:

- Overview
- Operations
- Channels
- Agents
- Workspace
- Chat
- Settings

Legacy detail pages remain as internal views, but they are moved behind section-level subnavigation instead of being exposed as equal top-level destinations.

## Section Mapping

### Overview

Purpose: first-screen health and situational awareness.

Content:

- gateway connection and auth state
- critical system status cards
- channel/session/cron/log summary
- active issues / degraded states / follow-up pointers
- quick links into detailed operations pages

This page should answer:

- Is the gateway healthy?
- What is broken or stale?
- Where should I click next?

### Operations

Purpose: day-to-day remote control and diagnostics.

Subsections:

- Sessions
- Usage
- Cron
- Approvals
- Logs
- Debug

These tools are operationally related and should feel like one workspace rather than six unrelated top-level tabs.

### Channels

Purpose: messaging ingress/egress health and setup.

This remains its own top-level area because channel state is one of the first things an operator checks remotely.

### Agents

Purpose: agent-specific runtime surfaces.

Subsections:

- Agent overview/files
- Skills
- Nodes / paired devices

This consolidates the current agent + skills + nodes sprawl under one operational domain.

### Workspace

Purpose: slower-moving content and knowledge surfaces.

Subsections:

- Projects
- Memory
- Documents
- Calendar

This becomes a secondary domain, not a first-line operational navigation cluster.

### Chat

Purpose: emergency/fallback direct interaction channel.

Chat remains accessible, but its visual prominence and default centering are reduced.

### Settings

Purpose: configuration and system administration.

Subsections:

- Config
- Instances / presence

If logs/debug are already covered in Operations, Settings should stay narrow and administrative.

## Navigation Design

- Replace the current multi-group collapsible nav with a smaller, flatter operations-first sidebar.
- Use a single clear primary section list instead of several competing taxonomies.
- Preserve direct deep-link support from old paths by mapping legacy routes onto their new parent section and selecting the matching subsection.
- Keep the sidebar desktop-first and optimized for rapid scanning.

## Page Structure Rules

Each major page should follow the same hierarchy:

1. Section title + one-sentence purpose
2. Summary strip or status cards
3. Subsection switcher
4. Detailed content area

This removes the current “everything is a card” feel and makes information density more predictable.

## Overview Design

The redesigned Overview page should contain:

- Primary health strip
  - gateway connection
  - auth mode
  - last refresh / freshness
  - critical counts
- Operational summary cards
  - channels
  - sessions
  - cron
  - approvals
  - logs/debug signals
- Degraded state / attention area
  - surfaced errors
  - stale integrations
  - actionable next clicks
- Minimal connection details
  - connection URL / token state / device trust, but trimmed so setup no longer dominates the page

## Visual Direction

- Treat the app as an operations console, not a chat shell.
- Reduce duplicated titles, subtitles, and labels.
- Use stronger hierarchy and spacing between summary and detail.
- Keep chat-specific controls localized to the Chat page.
- Prefer concise cards with obvious status semantics over sprawling setup blocks.

## Route Compatibility

Existing URLs like `/sessions`, `/usage`, `/cron`, `/skills`, `/nodes`, `/projects`, `/memory`, `/documents`, `/calendar`, `/debug`, `/logs`, and `/instances` should continue to work, but should resolve into the new parent tab plus the appropriate subsection.

## State Model Changes

Add persistent subsection selection state for:

- operations
- agents
- workspace
- settings

This allows the app to remember the last detailed view inside each section without needing many top-level tabs.

## Implementation Outline

1. Simplify navigation model and tab routing.
2. Add section/subsection state.
3. Rebuild Overview as an operations dashboard.
4. Create consolidated section renderers for Operations, Agents, Workspace, and Settings.
5. Map old routes to new section + subsection combinations.
6. Trim duplicated labels and move chat-only controls out of non-chat views.

## Testing

- navigation route normalization and backward-compatible path mapping
- tab/subsection switching
- overview rendering under connected/disconnected/degraded states
- settings persistence for new subsection state
- smoke checks for legacy deep links

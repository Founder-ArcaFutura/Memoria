# Shared Memory Spaces (Plain-Language Guide)

Memoria lets you decide whether a memory should live in your own private space or be shared with a wider group. The "team" (or "workspace") feature is how you create those shared spaces. Think of it as a digital filing cabinet: each drawer belongs to a group, and everyone in that group can drop in new notes or look up what is already there.

## Why you might use it

- **Work together without losing context.** A support team can collect troubleshooting steps in one place so the next person sees what has already been tried.
- **Keep private thoughts separate.** Anything you add while no team is active stays in your personal drawer.
- **Avoid accidental oversharing.** You explicitly decide when a note should land in the shared drawer (by joining a team or turning on automatic sharing for that team).
- **Audit who can see what.** Each team lists the people or agents who are allowed to read or write to it.

## What happens behind the scenes

1. **You (or an administrator) create a team.** Give it a simple name like `research-alliance`, optionally invite members, and decide whether new notes should be shared by default.
2. **You activate the team when you need it.** Once active, memories you mark as "share with the team" go into that shared drawer instead of your personal one.
3. **Members see the same shared history.** Everyone on the team can search and retrieve the shared notes, so planning sessions, decisions, and updates stay aligned.
4. **Turn it off to go solo.** Clear the active team to resume writing only to your private space.

## A quick story

> Lee leads a product trio with Ravi (design) and Mina (engineering). They create a `launch` team in Memoria and invite each other. During stand-ups they activate the `launch` team so meeting summaries, customer feedback, and action items are shared automatically. Later, Lee deactivates the team before jotting personal reflections so they stay private.

## How to try it

- **From configuration:** Set `memory.team_memory_enabled = true` (or choose a `team_mode` like `optional`) in your settings file so Memoria loads the collaborative features.
- **From the SDK:**
  ```python
  memoria = Memoria(team_memory_enabled=True, user_id="lee")
  memoria.enable()

  memoria.register_team_space(
      team_id="launch",
      members=["lee", "ravi", "mina"],
      share_by_default=True,
  )

  memoria.set_active_team("launch")
  memoria.store_memory(
      text="Summarised the customer interview.",
      share_with_team=True,
  )
  ```
- **From the CLI:**
  ```bash
  memoria teams create launch --member lee --member ravi --member mina --activate
  memoria teams list
  memoria teams clear  # go back to personal mode
  ```

## Where to learn more

- Deep dive into the concepts in [Teams & Workspaces](../core-concepts/teams-and-workspaces.md).
- Review the full feature set in [Team-aware deployments](../open-source/features.md#team-and-workspace-modes).
- Explore API and CLI options in [CLI Teams commands](../cli/index.md#teams).

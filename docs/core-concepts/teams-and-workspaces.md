# Teams & Workspaces

Memoria includes robust support for multi-tenancy through a flexible system of Teams and Workspaces. This allows multiple users or agents to collaborate within shared memory spaces while maintaining data isolation and privacy.

!!! tip "Need the plain-language version?"
    Start with the [Shared Memory Spaces guide](../getting-started/shared-memory-plain-language.md) for a story-driven overview before diving into the full technical model below.

## Core Concepts

-   **Namespace**: The fundamental unit of memory isolation. By default, all memories are stored within a single namespace (e.g., `"default"`).
-   **Team**: A group of users who can share a common memory namespace. This is the primary mechanism for collaboration.
-   **Workspace**: A more abstract grouping that functions as an alias for a team. It's designed to map to product-level concepts (e.g., a "project" or a "case file") and provides a way to manage shared memory spaces that might involve different groups of users over time.

Functionally, Workspaces are built on top of the Team system. When you create a workspace, you are creating a team with a specific namespace and set of members.

## How it Works

When you initialize Memoria, you can enable team and workspace functionality. The `StorageService` then resolves the correct namespace for each memory operation based on the active team or workspace.

-   **Personal Memory**: If no team or workspace is active, memories are stored in the user's personal namespace.
-   **Shared Memory**: When a team/workspace is active, memories can be stored in that team's shared namespace, making them accessible to all members of that team.

### Namespace Resolution

Memoria uses the following logic to determine where to store a memory:

1.  **Explicit Namespace**: If a `namespace` is provided directly in a function call, it is used.
2.  **Active Team/Workspace**: If a team or workspace is active, and the `share_with_team` flag is set, the memory is stored in the team's dedicated namespace (e.g., `"team:research-alliance"`).
3.  **Default Namespace**: If neither of the above applies, the memory is stored in the user's default personal namespace.

### Membership and Permissions

-   You can define `members` and `admins` for each team/workspace.
-   By default, Memoria enforces membership, meaning a user must be a member of a team to access its shared memory. This can be configured.
-   The `share_by_default` option on a team can be set to automatically store memories in the shared namespace when that team is active.

## Managing Teams & Workspaces

You can manage teams and workspaces programmatically using the `Memoria` SDK or via the [Command-Line Interface](./cli.md#teams).

### Example SDK Usage

```python
from memoria import Memoria

# Initialize Memoria with team support
memoria = Memoria(team_memory_enabled=True, user_id="alice")
memoria.enable()

# Create a new team
memoria.register_team_space(
    team_id="research-alliance",
    display_name="Research Alliance",
    members=["alice", "bob"],
    admins=["alice"]
)

# Activate the team for the current session
memoria.set_active_team("research-alliance")

# This memory will be stored in the "team:research-alliance" namespace
# and will be visible to both Alice and Bob.
memoria.store_memory(
    anchor="project-alpha-kickoff",
    text="We started Project Alpha today.",
    tokens=6,
    share_with_team=True
)
```

This system provides a powerful and flexible way to manage memory in complex, multi-user environments, making it suitable for enterprise applications, collaborative agents, and more.
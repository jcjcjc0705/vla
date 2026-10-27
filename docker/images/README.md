# Reference Dockerfiles

**These are snapshots, not the build source.** They are copied here so that the environment this
project depends on is readable without access to the private registry it is published to. The
authoritative versions live in their own repositories; if the two ever disagree, those win.

| File | Snapshot of | As of |
|---|---|---|
| `omx_bridge_image.Dockerfile` | `omx_bridge_image` | commit `d7b8b42`, 2026-08-07 |
| `omx_vla_image.Dockerfile` | `omx_vla_image` | commit `efef4f3`, 2026-08-10 |

```
sim_real_bridge_image     robot-agnostic sync engine; ships no robot data
└── omx_bridge_image      this arm: profile, USD/URDF/meshes, MoveIt, operator tools   (numpy 1.x)
    └── omx_vla_image     + torch cu128, LeRobot, GR00T, TensorBoard                   (numpy 2.x)
```

Both `FROM` lines point at a private registry, so neither builds as-is outside the original
environment. What they are useful for is the contract: read them alongside
[docs/SETUP.md](../../docs/SETUP.md#what-the-base-image-must-provide) to see exactly what the task
layer expects to find in its base image.

Two things in these files are worth reading even if you never build them, because both were expensive
to learn and neither is obvious:

- **`omx_vla_image` splits the torch install across six layers.** Not cosmetic — a single-layer install
  produces one 7.1 GB layer, and pushing a blob that large outlived the registry's auth token. Every
  smaller layer uploaded fine; that one retried five times and failed with
  `unauthorized: authentication required`. The CUDA runtime packages are therefore installed first,
  pinned and with `--no-deps`, so the final `torch` install finds its dependencies satisfied.
- **The build-time guard asserts the cu128 build survived dependency resolution**, because something
  quietly replacing it with a default-index wheel is invisible until a training run dies on a Blackwell
  card. ⚠️ The guard deliberately does **not** check MoveIt: `moveit_py` imports cleanly and segfaults
  at run time under numpy 2.x, so an import assertion there produces false confidence — which is
  exactly what happened once.

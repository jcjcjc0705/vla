"""Start Isaac's SimulationApp.

Must be called **before** importing anything else from ``isaacsim`` -- the
extensions that provide those modules only load once the app exists. Every entry
point therefore begins with ``app = start()`` and imports scene.py afterwards,
which is why the imports in those files look out of order.

Unlike build_scene.py, anything using this does take a GPU and will contend with
another Isaac session on the same machine.
"""
from __future__ import annotations


def start(headless: bool = True, livestream: bool = False):
    """Create the SimulationApp. ``livestream`` serves it over WebRTC."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless or livestream})
    if livestream:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.services.livestream.nvcf")
    return app

"""Runtime compatibility helpers for Isaac Sim headless launches."""


def enable_omni_ui(simulation_app, update_count=8):
    """Enable omni.ui after AppLauncher has started the Kit app."""
    import omni.kit.app

    app = omni.kit.app.get_app()
    extension_manager = app.get_extension_manager()

    extensions = extension_manager.get_extensions()
    extension_id = None
    for extension in extensions.values() if isinstance(extensions, dict) else extensions:
        candidate = extension.get("id") if isinstance(extension, dict) else str(extension)
        if candidate is None:
            continue
        if candidate == "omni.ui" or candidate.startswith("omni.ui-"):
            extension_id = candidate
            break

    if extension_id is None:
        raise RuntimeError(
            "Could not find the omni.ui extension. Expected an extension id "
            "named 'omni.ui' or starting with 'omni.ui-'."
        )

    extension_manager.set_extension_enabled_immediate(extension_id, True)
    for _ in range(update_count):
        simulation_app.update()

    import omni.ui  # noqa: F401

    print(f"Enabled Isaac Sim extension: {extension_id}", flush=True)
    return extension_id

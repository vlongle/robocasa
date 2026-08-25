import os

_configured_assets_root = os.environ.get("ROBOCASA_ASSETS_ROOT")
assets_root = _configured_assets_root or os.path.join(
    os.path.dirname(__file__), "assets"
)
if not os.path.isdir(assets_root):
    raise FileNotFoundError(f"RoboCasa assets root does not exist: {assets_root}")

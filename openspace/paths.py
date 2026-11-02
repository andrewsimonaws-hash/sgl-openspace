"""Where the local OpenSpace lives, read from the same environment variables openspace.cfg uses.

    OPENSPACE_ROOT   the OpenSpace folder that holds openspace.cfg, e.g. a 0.22 release
    OPENSPACE_USER   its user folder, when not OPENSPACE_ROOT/user (openspace.cfg reads it too)
    OPENSPACE_SRC    the OpenSpace source tree, for install_module.py only

Only the scripts that read from or write into OpenSpace need these. Everything inside the
repository uses paths relative to the scripts.
"""
import os
import sys


def _env(name, what):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set the {name} environment variable to {what} (see 'Run it' in the README).")
    return value


def root():
    return _env("OPENSPACE_ROOT", "the OpenSpace folder that holds openspace.cfg")


def user():
    return os.environ.get("OPENSPACE_USER") or os.path.join(root(), "user")


def src():
    return _env("OPENSPACE_SRC", "the OpenSpace source tree")


def assets():
    return os.path.join(user(), "data", "assets")


def profiles():
    return os.path.join(user(), "data", "profiles")


def recordings():
    return os.path.join(user(), "recordings")


def screenshots():
    return os.path.join(user(), "screenshots")


def mono_font():
    """OpenSpace's "Mono" font, the default for screen text and labels."""
    return os.path.join(root(), "data", "fonts", "Bitstream-Vera-Sans-Mono", "VeraMono.ttf")

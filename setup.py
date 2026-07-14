import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, PlatformError

class OptionalBuildExt(build_ext):
    """A build_ext subclass that falls back to pure Python if compilation fails."""
    def run(self):
        try:
            super().run()
        except (CCompilerError, PlatformError, ExecError) as e:
            print("*" * 60)
            print(f"WARNING: C compilation failed: {e}")
            print("Continuing installation in pure Python mode.")
            print("*" * 60)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except (CCompilerError, PlatformError, ExecError) as e:
            print("*" * 60)
            print(f"WARNING: C compilation of extension {ext.name} failed: {e}")
            print("Continuing installation in pure Python mode.")
            print("*" * 60)
            self.extensions = []

# Optional Cython compilation
ext_modules = []
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [
            Extension("gym_locm.engine.card", ["gym_locm/engine/card.py"]),
            Extension("gym_locm.engine.player", ["gym_locm/engine/player.py"]),
            Extension("gym_locm.engine.action", ["gym_locm/engine/action.py"]),
            Extension("gym_locm.engine.phases", ["gym_locm/engine/phases.py"]),
            Extension("gym_locm.engine.game_state", ["gym_locm/engine/game_state.py"]),
        ],
        compiler_directives={"language_level": "3", "boundscheck": False},
    )
except ImportError:
    print("Cython not found. Installing in pure Python mode.")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
)

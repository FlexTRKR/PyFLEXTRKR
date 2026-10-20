"""
Import checks of pyflextrkr.visualization that need no plotting package: the CI environment installs `dask` but not `distributed`, so
the module of the file loop (runner.py) must import without dask.distributed (it is imported only for the parallel mode), and importing
the package must not import matplotlib or cartopy (the tracking never needs them). Each check runs in a new interpreter, so what
other tests imported before does not matter.
"""
import subprocess
import sys


def run_python(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)


def test_the_runner_imports_without_the_distributed_package():
    # `sys.modules[name] = None` makes `import name` fail like a package that is not installed
    result = run_python("import sys; sys.modules['distributed'] = None; import pyflextrkr.visualization.runner")
    assert result.returncode == 0, result.stderr


def test_the_serial_loop_runs_without_the_distributed_package():
    code = ("import sys; sys.modules['distributed'] = None; "
            "from pyflextrkr.visualization.runner import run_frames; "
            "done = []; "
            "work = lambda f, t, m, p, c, i: done.append(f); "
            "run_frames(work, ['a', 'b'], {}, {}, {}, {}, run_parallel=0); "
            "assert done == ['a', 'b'], done")
    result = run_python(code)
    assert result.returncode == 0, result.stderr


def test_importing_the_package_does_not_import_matplotlib_or_cartopy():
    result = run_python("import sys; import pyflextrkr.visualization; "
                        "assert 'matplotlib' not in sys.modules and 'cartopy' not in sys.modules, sorted(m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'cartopy'))[:5]")
    assert result.returncode == 0, result.stderr

def malis(
        affs,
        gt,
        force_rebuild = False):

    import sys, os
    import shutil
    import glob
    import numpy
    import fcntl

    try:
        import hashlib
    except ImportError:
        import md5 as hashlib

    from distutils.core import Distribution, Extension
    from distutils.command.build_ext import build_ext
    from distutils.sysconfig import get_config_vars, get_python_inc

    import Cython
    from Cython.Compiler.Main import Context, default_options
    from Cython.Build.Dependencies import cythonize

    source_dir = os.path.dirname(os.path.abspath(__file__))
    source_files = [
            os.path.join(source_dir, 'frontend.pyx'),
            os.path.join(source_dir, 'malis_loss_layer.hpp'),
            os.path.join(source_dir, 'malis_loss_layer.cpp'),
    ]
    source_files.sort()
    source_files_hashes = [ hashlib.md5(open(f, 'r').read().encode('utf-8')).hexdigest() for f in source_files ]

    key = source_files_hashes, sys.version_info, sys.executable, Cython.__version__
    module_name = 'pymalis_' + hashlib.md5(str(key).encode('utf-8')).hexdigest()
    lib_dir=os.path.expanduser('~/.cython/inline')

    # since this could be called concurrently, there is no good way to check
    # whether the directory already exists
    try:
        os.makedirs(lib_dir)
    except:
        pass

    # make sure the same module is not build concurrently
    with open(os.path.join(lib_dir, module_name + '.lock'), 'w') as lock_file:
        fcntl.lockf(lock_file, fcntl.LOCK_EX)

        try:

            if lib_dir not in sys.path:
                sys.path.append(lib_dir)
            if force_rebuild:
                raise ImportError
            else:
                __import__(module_name)

            print("Re-using already compiled pymalis version")

        except ImportError:

            print("Compiling pymalis in " + str(lib_dir))

            cython_include_dirs = ['.']
            ctx = Context(cython_include_dirs, default_options)

            scoring_function_include_dir = os.path.join(lib_dir, module_name)
            if not os.path.exists(scoring_function_include_dir):
                os.makedirs(scoring_function_include_dir)

            include_dirs = [
                source_dir,
                os.path.dirname(get_python_inc()),
                numpy.get_include(),
            ]

            # cython requires that the pyx file has the same name as the module
            shutil.copy(
                    os.path.join(source_dir, 'frontend.pyx'),
                    os.path.join(lib_dir, module_name + '.pyx')
            )
            shutil.copy(
                    os.path.join(source_dir, 'malis_loss_layer.cpp'),
                    os.path.join(lib_dir, module_name + '_malis_loss_layer.cpp')
            )

            # Remove the "-Wstrict-prototypes" compiler option, which isn't valid 
            # for C++.
            cfg_vars = get_config_vars()
            if "CFLAGS" in cfg_vars:
                cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

            extension = Extension(
                    module_name,
                    sources = [
                        os.path.join(lib_dir, module_name + '.pyx'),
                        os.path.join(lib_dir, module_name + '_malis_loss_layer.cpp')
                    ],
                    include_dirs=include_dirs,
                    language='c++',
                    extra_link_args=['-std=c++11'],
                    extra_compile_args=['-std=c++11', '-w']
            )
            build_extension = build_ext(Distribution())
            build_extension.finalize_options()
            build_extension.extensions = cythonize([extension], quiet=True, nthreads=2)
            build_extension.build_temp = lib_dir
            build_extension.build_lib  = lib_dir
            build_extension.run()

    return __import__(module_name).malis(affs, gt)

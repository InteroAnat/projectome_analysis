# XXX: HOST == Linux/macOS x86_64

# XXX: avx2 <__floatdidf> <__floatdihf> <__floatsihf> <__floatsisf>

using BinaryBuilder, Pkg

name = "Gapr"
version = v"1.3.0"

### Tempbranch
import LibGit2
hash = LibGit2.with(LibGit2.GitRepo(joinpath(homedir(), "Desktop/gapr/.git"))) do r
    c = LibGit2.GitCommit(r, "julia")
    string(LibGit2.GitHash(c))
end
@warn "hash $hash"

# Collection of sources required to complete build
sources = [
    GitSource("git://localhost/gapr.git", hash),
    ArchiveSource("https://ziglang.org/download/0.14.0/zig-linux-x86_64-0.14.0.tar.xz", "473ec26806133cf4d1918caf1a410f8403a13d979726a9045b421b685031a982"),
    ### DirectorySource("./bundled")
]

# Bash recipe for building across all platforms
script = raw"""
#!/bin/bash

export ZIG_GLOBAL_CACHE_DIR=$WORKSPACE/zigcache
ZIG=$WORKSPACE/srcdir/zig-linux-x86_64-*/zig
sysroot=/opt/${target}/${target}/sys-root

export PKG_CONFIG_PATH=/workspace/destdir/bin/pkgconfig:/workspace/destdir/lib/pkgconfig:/workspace/destdir/share/pkgconfig
#
# find /opt /workspace -name '*crypto*' >&2 
# pkg-config --list-all >&2
# #find /opt /workspace -name '*vpx*' >&2 
# find /opt /workspace -name '*boost_ios*' >&2 
# #exit -1
# #find /opt /workspace -name '*gcc*' >&2 

# NOTE: use newer glibc
if [[ -f "$prefix/usr/include/sys/single_threaded.h" ]]; then
    GLIBC_ARTIFACT_DIR=$(dirname $(dirname $(dirname $(dirname $(realpath "$prefix/usr/include/sys/single_threaded.h")))))
    rsync --archive ${GLIBC_ARTIFACT_DIR}/ $sysroot/
fi
if [[ -f "$prefix/bin/liblmdb.dll" ]]; then
    {
        echo LIBRARY liblmdb.dll
        echo EXPORTS
        echo mdb_cursor_close
        echo mdb_cursor_get
        echo mdb_cursor_open
        echo mdb_cursor_put
        echo mdb_dbi_open
        echo mdb_env_close
        echo mdb_env_create
        echo mdb_env_open
        echo mdb_env_set_flags
        echo mdb_env_set_mapsize
        echo mdb_env_sync
        echo mdb_txn_abort
        echo mdb_txn_begin
        echo mdb_txn_commit
    } > $(dirname $ZIG)/lib/libc/mingw/lib-common/liblmdb.def
fi

case "${target}" in
        # XXX: Zig supports glibc abi >=2.32; toolchain glibc too old.
    x86_64-linux-gnu) zig_tgt="x86_64-linux-gnu.2.32" ;;
    x86_64-w64-mingw32) zig_tgt="x86_64-windows-gnu" ;;
    x86_64-apple-darwin*) zig_tgt="x86_64-macos" ;;
    *) echo "Unsupported target $target" >&2; exit -1 ;;
esac

pushd gapr
$ZIG build  --prefix $prefix \
    --prefix-exe-dir $bindir --prefix-lib-dir $libdir --prefix-include-dir $includedir \
    -Dtarget="$zig_tgt" -Dcpu=${zig_tgt%%-*} -Dbb_target=$bb_target \
    --sysroot / --system $prefix \
    -Dgui=false \
    --verbose-cc --verbose-link --verbose-llvm-cpu-features
popd
"""

platforms = [
    Platform("x86_64", "linux"; libc="glibc", cxxstring_abi="cxx11");
    # Platform("x86_64", "windows")
    # Platform("x86_64", "macos")
]
isglibc(p) = libc(p) == "glibc"

products = [
    LibraryProduct(["libgapr-core", "gapr-core"], :libgaprcore)
    ExecutableProduct("gapr-export", :gaprexport)
]

dependencies = [
    Dependency("dlfcn_win32_jll", platforms=filter(Sys.iswindows, platforms))
    Dependency("Zlib_jll")
    Dependency("LMDB_jll")
    Dependency("Libtiff_jll")
    BuildDependency("LibVPX_jll")
    Dependency("LibCURL_jll", compat="7.73,8")
    Dependency("OpenSSL_jll")
    Dependency("boost_jll", compat="=1.87.0")
    # BuildDependency("msgpack_cxx_jll")
]

# XXX: julia 1.7 bundles libstdc++ abi 3.4.29; gcc >11 fails.
preferred_gcc_version = v"11.1.0"
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; julia_compat="1.6", preferred_gcc_version)

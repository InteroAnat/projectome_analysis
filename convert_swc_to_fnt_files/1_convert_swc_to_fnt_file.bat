@echo off
setlocal enabledelayedexpansion

set "data_dir=D:\BaiduSyncdisk\macaque-PFC-fmost\1-3_convert_swc_to_fnt_files\swc_files_no_flip_axon_qroot"

for %%a in ("%data_dir%") do (
    set "parent_dir=%%~dpa"
)
set "parent_dir=!parent_dir:~0,-1!"
set "fnt_dir=!parent_dir!\fnt_files_no_flip_axon_qroot"

if not exist "!fnt_dir!" (
    mkdir "!fnt_dir!"
    echo Created output directory: !fnt_dir!
)

for %%f in ("%data_dir%\*.swc") do (
    set "swc_file=%%f"
    call set "file_base=%%~nf"

    echo Output FNT file: !fnt_dir!\!file_base!.fnt

    fnt-from-swc "!swc_file!" "!fnt_dir!\!file_base!.fnt"
)

endlocal

#!/bin/bash

# 设置 SWC 文件所在目录
data_dir=/run/media/admin/fedora_localhost-live/home/admin/gapr/tmp_2/swc_files_flip_axon_qroot

# 获取父目录路径
parent_dir=$(dirname "$data_dir")

# 构建 FNT 文件保存目录
fnt_dir="${parent_dir}/fnt_files_flip_axon_qroot"

# 创建保存目录（如果不存在）
mkdir -p "$fnt_dir"
echo "FNT output directory: $fnt_dir"

# 避免空格/特殊字符造成遍历问题
oldIFS=$IFS
IFS=$(echo -en "\n\b")

# 遍历所有 .swc 文件并转换
for swc_file in $(ls "$data_dir"/*.swc); do
    file_name=$(basename "$swc_file" .swc)
    output_fnt="${fnt_dir}/${file_name}.fnt"

    echo "Processing SWC file: $swc_file"
    echo "Output FNT file will be: $output_fnt"

    fnt-from-swc "$swc_file" "$output_fnt"
done

# 恢复 IFS
IFS=$oldIFS

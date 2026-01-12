#!/bin/bash

home_dir="/home/admin/macaque-neuron-codes/2-2_get_fnt-dist_matrix"
data_dir="$home_dir/swc_files_flip_axon_qroot_ds-node-branch_test"

fnt_component_dir="/home/admin/macaque-neuron-codes/fnt-19-9.fedora/usr/local/bin"

oldIFS=$IFS
IFS=$(echo -en "\n\b")

for swc_file in `ls $data_dir/*.swc`
do
    file_name=${swc_file%.swc*}
    $fnt_component_dir/fnt-from-swc $swc_file $file_name.fnt
    $fnt_component_dir/fnt-decimate -d 5000 -a 5000 $file_name.fnt $file_name.decimate.fnt
done

IFS=$oldIFS




# home_dir="/home/admin/macaque-neuron-codes/2-2_get_fnt-dist_matrix"
# data_dir="$home_dir/swc_files_flip_axon_qroot_ds-node-branch_test"

# fnt_component_dir="/home/admin/macaque-neuron-codes/fnt-19-9.fedora/usr/local/bin"

# export data_dir

# parallel_process() {
#     swc_file=$1
#     file_name=${swc_file%.swc*}
#     $fnt_component_dir/fnt-from-swc $swc_file $file_name.fnt
#     $fnt_component_dir/fnt-decimate -d 5000 -a 5000 $file_name.fnt $file_name.decimate.fnt
# }

# export -f parallel_process

# find $data_dir -name "*.swc" | parallel parallel_process {}

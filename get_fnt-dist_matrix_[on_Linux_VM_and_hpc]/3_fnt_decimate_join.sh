#!/bin/bash

home_dir="/home/admin/macaque-neuron-codes/2-2_get_fnt-dist_matrix"
data_dir="$home_dir/swc_files_flip_axon_qroot_ds-node-branch_test"

fnt_component_dir="/home/admin/macaque-neuron-codes/fnt-19-9.fedora/usr/local/bin"

oldIFS=$IFS
IFS=$(echo -en "\n\b")

$fnt_component_dir/fnt-join $data_dir/*.decimate.fnt -o "$home_dir/fnt_decimate_join.fnt"

IFS=$oldIFS

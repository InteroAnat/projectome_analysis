#!/bin/bash

code_directory="/home/admin/macaque-neuron-codes/2-2_get_fnt-dist_matrix/000_codes"

bash $code_directory/1_convert_swc_decimate_to_fnt_decimate.sh

Rscript $code_directory/2_change_fnt_decimate_neuron_name.R

bash $code_directory/3_fnt_decimate_join.sh

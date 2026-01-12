
library(parallel)
library(foreach)
library(doParallel)
library(data.table)

home_directory <- 'D:/BaiduSyncdisk/macaque-PFC-fmost/2-2_get_fnt-dist_matrix/'
original_result_dir_name <- 'swc_files_flip_axon_qroot_ds-node-branch_test'

work_directory <- paste0(home_directory, original_result_dir_name, '/')

core_num <- detectCores()  ### 设置多线程使用的CPU核心数

##########
all_fnt_decimate_file_names <- dir(work_directory, pattern = '\\.decimate\\.fnt$')


###################
cl <- makeCluster(core_num)
registerDoParallel(cl)

change_fnt_decimate_neuron_name <- foreach(iName = 1:length(all_fnt_decimate_file_names), .packages = 'data.table') %dopar% {
  each_fnt_decimate_file_name <- all_fnt_decimate_file_names[iName]
  each_neuron_name <- sub('(.*)\\.decimate\\.fnt', '\\1', each_fnt_decimate_file_name)
  each_fnt_decimate_content <- fread(paste0(work_directory, each_fnt_decimate_file_name), sep = "\n", header = FALSE)[[1]]
  each_fnt_decimate_content[length(each_fnt_decimate_content)] <- paste0('0 ', each_neuron_name)
  
  writeLines(each_fnt_decimate_content, con = paste0(work_directory, each_fnt_decimate_file_name), sep = '\n')
}

stopCluster(cl)



# ###################
# for (each_fnt_decimate_file_name in all_fnt_decimate_file_names) {
#   each_neuron_name <- sub('(.*)\\.decimate\\.fnt', '\\1', each_fnt_decimate_file_name)
#   each_fnt_decimate_content <- fread(paste0(work_directory, each_fnt_decimate_file_name), sep = "\n", header = FALSE)[[1]]
#   each_fnt_decimate_content[length(each_fnt_decimate_content)] <- paste0('0 ', each_neuron_name)
# 
#   writeLines(each_fnt_decimate_content, con = paste0(work_directory, each_fnt_decimate_file_name), sep = '\n')
# }








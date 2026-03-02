
library(data.table)
library(foreach)
library(doParallel)
library(parallel)

home_directory <- 'D:/BaiduSyncdisk/macaque-PFC-fmost/1-3_convert_swc_to_fnt_files/'

original_fnt_file_directory <- paste0(home_directory, 'fnt_files_flip_axon_qroot', '/')
# original_fnt_file_directory <- paste0(home_directory, 'fnt_files_no_flip_axon_qroot', '/')


##################
all_fnt_file_names <- dir(original_fnt_file_directory, pattern = '\\.fnt$')

# 并行设置
num_cores <- parallel::detectCores() - 1
cl <- parallel::makeCluster(num_cores)
doParallel::registerDoParallel(cl)

# 并行处理
invisible(
  foreach(each_fnt_file_name = all_fnt_file_names,
          .packages = c("data.table"),
          .inorder = FALSE) %dopar% {
            
            each_neuron_name <- sub('(.*)\\.fnt', '\\1', each_fnt_file_name)
            
            each_fnt_file_content <- data.table::fread(
              paste0(original_fnt_file_directory, each_fnt_file_name),
              sep = '\n', header = FALSE
            )[[1]]
            
            each_fnt_file_content[length(each_fnt_file_content)] <- paste0('0 ', each_neuron_name)
            
            writeLines(
              each_fnt_file_content,
              con = paste0(original_fnt_file_directory, each_fnt_file_name),
              sep = '\n'
            )
            
            NULL
          }
)

# 释放集群
parallel::stopCluster(cl)












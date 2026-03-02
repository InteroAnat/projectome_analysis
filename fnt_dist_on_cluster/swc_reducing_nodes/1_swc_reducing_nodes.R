
library(nat)
library(stringi)

home_directory <- 'D:/BaiduSyncdisk/macaque-PFC-fmost/1-2_swc_reducing_nodes/'

original_swc_directory <- paste0(home_directory, 'swc_files_flip_axon_qroot', '/')

swc_file_directory_new <- paste0(home_directory, 'swc_files_flip_axon_qroot_ds-node', '/')
if (!dir.exists(swc_file_directory_new)) {dir.create(swc_file_directory_new)}

######
all_neuron_swc_names <- dir(original_swc_directory, pattern = '\\.swc$')
all_neuron_names <- substr(all_neuron_swc_names, 1, nchar(all_neuron_swc_names)-4)

all_neuron_swc_names_new <- all_neuron_swc_names

sampling_factor <- 6    ### 设置砍减节点的倍数【需尝试砍减多少倍合适，一般可以视情况设置为3-7】


############
all_neurons_original <- read.neurons(paths = paste0(original_swc_directory, all_neuron_swc_names), neuronnames = all_neuron_names)

## ---- 并行设置 ----
# core_num <- parallel::detectCores()
core_num <- 4    ### 下面的程序占用内存较大，因此这里的CPU核心数不宜设置过高！
cl <- makeCluster(core_num)
registerDoParallel(cl)

# 并行执行：等价于原先的 for (iNeuron in 1:length(all_neuron_swc_names)) { ... }
invisible(
  foreach(iNeuron = seq_along(all_neuron_swc_names),
          .packages = c("nat", "stringi")) %dopar% {
            
            each_neuron <- all_neurons_original[[iNeuron]]
            each_neuron_node_info_df <- each_neuron$d
            each_neuron_all_seglist <- each_neuron$SegList
            
            ######
            each_neuron_all_seg_node_info_list_new <- list()
            
            for (each_seglist in each_neuron_all_seglist) {
              sampling_indices <- seq(from = 1, to = length(each_seglist), by = sampling_factor)
              sampled_row_indices <- each_seglist[sampling_indices]
              if (sampled_row_indices[length(sampled_row_indices)] != each_seglist[length(each_seglist)]) {
                sampled_row_indices <- append(sampled_row_indices, each_seglist[length(each_seglist)])
              }
              
              each_seg_node_info_df_new <- each_neuron_node_info_df[sampled_row_indices, ]
              each_seg_node_info_df_new[2:nrow(each_seg_node_info_df_new), ]$Parent <-
                each_seg_node_info_df_new[1:(nrow(each_seg_node_info_df_new)-1), ]$PointNo
              
              each_neuron_all_seg_node_info_list_new <-
                append(each_neuron_all_seg_node_info_list_new, list(each_seg_node_info_df_new))
            }
            
            if (length(each_neuron_all_seg_node_info_list_new) > 1) {
              each_neuron_all_node_info_df_new <- do.call(rbind, each_neuron_all_seg_node_info_list_new)
            } else {
              each_neuron_all_node_info_df_new <- each_neuron_all_seg_node_info_list_new[[1]]
            }
            
            duplicated_point_indices <- which(duplicated(each_neuron_all_node_info_df_new$PointNo))
            if (length(duplicated_point_indices) > 0) {
              each_neuron_all_node_info_df_new <- each_neuron_all_node_info_df_new[-duplicated_point_indices, ]
            }
            
            rownames(each_neuron_all_node_info_df_new) <- 1:nrow(each_neuron_all_node_info_df_new)
            
            ######
            each_neuron_new <- as.neuron(each_neuron_all_node_info_df_new)
            each_neuronlist_new <- neuronlist()
            each_neuronlist_new <- append(each_neuronlist_new, neuronlist(each_neuron_new))
            names(each_neuronlist_new) <- all_neuron_swc_names_new[iNeuron]
            
            # 并行写入不同文件名，互不冲突
            write.neurons(nl = each_neuronlist_new,
                          dir = swc_file_directory_new,
                          files = all_neuron_swc_names_new[iNeuron],
                          Force = TRUE)
            
            NULL
          }
)

stopCluster(cl)









################################  在需要画图验证时使用
library(nat)

home_directory <- 'D:/BaiduSyncdisk/macaque-PFC-fmost/1-2_swc_reducing_nodes/'

original_swc_directory <- paste0(home_directory, 'swc_files_flip_axon_qroot', '/')
swc_file_directory_new <- paste0(home_directory, 'swc_files_flip_axon_qroot_ds-node', '/')


all_neuron_swc_names_new <- dir(swc_file_directory_new, pattern = '\\.swc$')
all_neuron_swc_names_new <- all_neuron_swc_names_new[1:5]    ### 选开头的5个
# all_neuron_swc_names_new <- all_neuron_swc_names_new[length(all_neuron_swc_names_new):(length(all_neuron_swc_names_new)-5)]    ### 选结尾的的（最简单的）5个
all_neuron_names_new <- substr(all_neuron_swc_names_new, 1, nchar(all_neuron_swc_names_new)-4)
all_neurons_new_plotted <- read.neurons(paths = paste0(swc_file_directory_new, all_neuron_swc_names_new), neuronnames = all_neuron_names_new)

all_neuron_swc_names <- all_neuron_swc_names_new
all_neuron_names <- all_neuron_names_new
all_neurons_plotted <- read.neurons(paths = paste0(original_swc_directory, all_neuron_swc_names), neuronnames = all_neuron_names)


brain_shell_obj_directory <- 'D:/BaiduSyncdisk/macaque-PFC-fmost/000_macaque_standard_brain/NMT_v2.0_sym_main/99999_root.obj'
brain_shell_obj <- readOBJ(brain_shell_obj_directory)


########
open3d()
# shade3d(brain_shell_obj,color="grey",override = T,alpha=0.05)

plot3d(all_neurons_plotted, color = 'red', soma = 200)
plot3d(all_neurons_new_plotted, color = 'blue', soma = 200)










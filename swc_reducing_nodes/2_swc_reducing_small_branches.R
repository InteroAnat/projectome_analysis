
library(nat)
library(rdist)
library(parallel)
library(foreach)
library(doParallel)

home_directory <- 'D:/BaiduSyncdisk/macaque-PFC-fmost/1-2_swc_reducing_nodes/'

original_swc_directory <- paste0(home_directory, 'swc_files_flip_axon_qroot_ds-node', '/')

swc_file_directory_new <- paste0(home_directory, 'swc_files_flip_axon_qroot_ds-node-branch', '/')
if (!dir.exists(swc_file_directory_new)) {dir.create(swc_file_directory_new)}

core_num <- detectCores() - 1    ###### 指定多线程用的CPU线程数


##############
snp_simplifyNeuronByFPS <- function(neuron, k) {
  tryCatch({
    fps <- rdist::farthest_point_sampling(
      mat = neuron$d[, c("X", "Y", "Z")],
      metric = "euclidean",
      k = k,
      initial_point_index = 1,
      return_clusters = TRUE
    )
    
    index_fps <- fps[[1]]
    label_sample <- matrix(nrow = nrow(neuron$d), ncol = 1, data = 0)
    
    for (iFPS in 1:k) {
      index_fps_node <- index_fps[iFPS]
      index_node_current <- index_fps_node
      
      while (TRUE) {
        parentID <- neuron$d$Parent[index_node_current]
        if (parentID == -1) break
        
        index_parentNode <- which(neuron$d$PointNo == parentID)
        index_node_current <- index_parentNode
        label_sample[index_parentNode, 1] <- 1
      }
    }
    
    # rgl::open3d()
    # rgl::points3d(x = neuron$d$X[which(label_sample == 1)],
    #               y = neuron$d$Y[which(label_sample == 1)],
    #               z = neuron$d$Z[which(label_sample == 1)])
    # 
    # rgl::points3d(x = neuron$d$X + 10,
    #               y = neuron$d$Y,
    #               z = neuron$d$Z, col = "red")
    
    swc_new <- neuron$d[which(label_sample == 1), ]
    neuron_new <- nat::as.neuron(swc_new)
    return(neuron_new)
    
  }, error = function(e) {
    # 检查是否是指定错误
    if (grepl("index_max\\(\\): object has no elements", conditionMessage(e))) {
      return(neuron)
    } else {
      stop(e) # 不是指定错误则抛出
    }
  })
}



##############
all_neuron_swc_file_names <- dir(original_swc_directory, pattern = '.swc')
all_neuron_names <- sub('(.*)\\.swc', '\\1', all_neuron_swc_file_names)
all_neurons <- read.neurons(paths = paste0(original_swc_directory, all_neuron_swc_file_names), neuronnames = all_neuron_names)


get_simplified_neuron <- function(neuron){
  kept_terminal_node_num <- 30    ### 在这里设置最终保留的末端点数目【需尝试砍减到多少个末端点合适，一般可以视情况设置为25-40】
  
  kept_terminal_node_num <- min(nrow(neuron$d), kept_terminal_node_num)
  neuron_simplified <- snp_simplifyNeuronByFPS(neuron, kept_terminal_node_num)
  return(neuron_simplified)
}



##############
## 并行设置
cl <- makeCluster(core_num)
registerDoParallel(cl)

# 并行写出
invisible(
  foreach(iNeuron = seq_along(all_neurons),
          .packages = c("nat", "rdist"),
          .export = c("get_simplified_neuron",
                      "snp_simplifyNeuronByFPS",
                      "swc_file_directory_new",
                      "all_neuron_names")) %dopar% {
                        each_neuron_name <- all_neuron_names[iNeuron]
                        each_neuron_simplified <- get_simplified_neuron(all_neurons[[iNeuron]])
                        each_neuronlist_simplified <- neuronlist(each_neuron_simplified)
                        names(each_neuronlist_simplified) <- each_neuron_name
                        nat::write.neurons(
                          nl = each_neuronlist_simplified,
                          dir = swc_file_directory_new,
                          files = paste0(each_neuron_name, ".swc"),
                          Force = TRUE
                        )
                        NULL
                      }
)

stopCluster(cl)





# ######
# cl <- makeCluster(core_num)
# registerDoParallel(cl)
# 
# all_neurons_simplified <- foreach(iNeuron = 1:length(all_neurons)) %dopar% {
#   get_simplified_neuron(all_neurons[[iNeuron]])
# }
# all_neurons_simplified <- as.neuronlist(all_neurons_simplified)
# names(all_neurons_simplified) <- all_neuron_swc_file_names
# 
# stopCluster(cl)
# 
# ##############
# write.neurons(nl = all_neurons_simplified, dir = swc_file_directory_new, files = all_neuron_swc_file_names, Force = TRUE)



# ######
# all_neurons_simplified <- lapply(all_neurons, get_simplified_neuron)
# all_neurons_simplified <- as.neuronlist(all_neurons_simplified)
# names(all_neurons_simplified) <- all_neuron_swc_file_names
# 
# 
# ######
# all_neurons_simplified <- neuronlist()
# 
# for (iNeuron in 1:length(all_neurons)) {
#   each_neuron <- all_neurons[[iNeuron]]
#   each_neuron_name <- all_neuron_names[iNeuron]
# 
#   each_neuron_simplified <- snp_simplifyNeuronByFPS(each_neuron, 30)
#   each_neuron_simplified_list <- neuronlist(each_neuron_simplified)
#   names(each_neuron_simplified_list) <- each_neuron_name
#   all_neurons_simplified <- append(all_neurons_simplified, each_neuron_simplified_list)
# }
# 
# ##############
# write.neurons(nl = all_neurons_simplified, dir = swc_file_directory_new, files = all_neuron_swc_file_names, Force = TRUE)














################################  在需要画图验证时使用
library(nat)

home_directory <- 'D:/BaiduSyncdisk/macaque-PFC-fmost/1-2_swc_reducing_nodes/'

original_swc_directory <- paste0(home_directory, 'swc_files_flip_axon_qroot', '/')
swc_file_directory_new <- paste0(home_directory, 'swc_files_flip_axon_qroot_ds-node-branch_2', '/')


all_neuron_swc_names_new <- dir(swc_file_directory_new, pattern = '\\.swc$')
all_neuron_swc_names_new <- all_neuron_swc_names_new[1:200]    ### 选开头的200个
# all_neuron_swc_names_new <- all_neuron_swc_names_new[length(all_neuron_swc_names_new):(length(all_neuron_swc_names_new)-5)]    ### 选结尾的（最简单的）5个
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
plot3d(all_neurons_plotted, color = 'red', soma = 80)
open3d()
plot3d(all_neurons_new_plotted, color = 'blue', soma = 80)

















library(readxl)
library(tidyverse)

m1_result <- "tables/251637_M1.xlsx"
m1_df <- read_excel(m1_result, sheet = "Projection_Strength_ipsi")

ins_results <- "tables/251637_INS.xlsx"
ins_df <- read_excel(ins_results, sheet = "Projection_Strength_ipsi")




table_path <- r"(D:\projectome_analysis\atlas\ARM_key_all.txt)"
global_id_df <- read_delim(table_path)







insula_abbr_table <- global_id_df %>%
  filter(str_detect(Full_Name, regex('insula', ignore_case = TRUE))) %>%
  pull(Abbreviation)

m1_abbr_table <- global_id_df %>%
  filter(str_detect(Abbreviation, regex('m1', ignore_case = TRUE))) %>% pull(Abbreviation)


clean_insula_table <- insula_abbr_table %>%
  str_remove_all("^(CL|CR|SL|SR)[-_]?") %>%  # Remove prefix + separator
  unique() %>%
  discard(~ .x == "")  
clean_m1_table <- m1_abbr_table %>%
  str_remove_all("^(CL|CR|SL|SR)[-_]?") %>%  # Remove prefix + separator
  unique() %>%
  discard(~ .x == "")  

insula_pattern <- paste(clean_insula_table, collapse = "|")
m1_pattern <- paste(clean_m1_table, collapse = "|")

m1_cols_in_ins <- names(ins_df)[str_detect(names(ins_df), m1_pattern)]

ins_cols_in_m1 <- names(m1_df)[str_detect(names(m1_df), insula_pattern)]

m1_insula_neurons <- m1_df %>%
  filter(if_any(all_of(ins_cols_in_m1), ~ .x != 0))




ins_m1_neurons <- ins_df %>%
  filter(if_any(all_of(m1_cols_in_ins), ~ .x != 0))




motor_abbr_table <- global_id_df %>%
  filter(str_detect(Full_Name, regex('motor', ignore_case = TRUE))) %>% pull(Abbreviation)

clean_motor_terms <- motor_abbr_table %>%
  str_remove_all("^(CL|CR|SL|SR)[-_]?") %>%  # Remove prefix + separator
  unique() %>%
  discard(~ .x == "") 
motor_pattern <- paste(clean_motor_table, collapse = "|")


motor_cols_in_ins <- names(ins_df)[str_detect(names(ins_df), motor_pattern)]
ins_motor_neurons <- ins_df %>%
  filter(if_any(all_of(motor_cols_in_ins), ~ .x != 0)) %>% select (all_of(motor_cols_in_ins))


library(ggplot2)
library(tidyr)
library(tibble)

# Convert to long format, preserving row names as a column
motor_long <- ins_motor_neurons %>%
  rownames_to_column("neuron_id") %>%    # Convert row names to column
  pivot_longer(cols = -neuron_id, 
               names_to = "motor_region", 
               values_to = "strength") %>%
  filter(strength > 0)

# Show all neuron IDs (if < ~50 neurons)
ggplot(motor_long, aes(x = motor_region, y = neuron_id, size = strength)) +
  geom_point(aes(color = strength), alpha = 0.7) +
  scale_color_gradient(low = "lightblue", high = "darkred") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 6),    # SHOW row names, small font
        axis.text.x = element_text(angle = 45, hjust = 1))

# If too many neurons: show only neurons with strong projections
top_neurons <- motor_long %>%
  group_by(neuron_id) %>%
  summarise(total_strength = sum(strength)) %>%
  top_n(20, total_strength) %>%
  pull(neuron_id)

motor_long_filtered <- motor_long %>% 
  filter(neuron_id %in% top_neurons)

ggplot(motor_long_filtered, aes(x = motor_region, y = neuron_id)) +
  geom_tile(aes(fill = strength), color = "white") +  # Heatmap style
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8))  # Clear row names

library(tidyverse)

table_path <- r"(D:\projectome_analysis\atlas\ARM_key_all.txt)"
global_id_df <- read_delim(table_path)
insula_abbr_table <- global_id_df %>%
  filter(str_detect(Full_Name, regex('insula', ignore_case = TRUE))) %>%
  pull(Abbreviation)


insula_abbr_table <- global_id_df %>%
  filter(str_detect(Full_Name, regex('insula', ignore_case = TRUE))) %>%
  pull(Abbreviation)



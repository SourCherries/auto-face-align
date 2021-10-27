library(tidyverse)
data <- read.csv("table-DLIB-bad-landmarks-NIM.csv")

dat <- as_tibble(data)

dat_wide <- dat %>%
  pivot_wider(names_from = Expression, values_from = Number)
  
# https://tidyr.tidyverse.org/reference/pivot_wider.html

write_csv(dat_wide, "table-wide-DLIB-bad-landmarks-NIM.csv")

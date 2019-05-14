library('tidyverse')

df <- read_csv('curves.csv')

ggplot(df, aes(x = step)) +
    geom_line(aes(y = svm_precision), color = 'red') +
    geom_line(aes(y = svm_recall), color = 'blue') +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
    labs(title = 'SVM Precision and Recall',
         x = 'Percentage Predicted as 1',
         y = 'Precision / Recall Value')
ggsave('plot.png')

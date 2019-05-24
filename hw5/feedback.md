# Feedback
* [x] When calling read_csv, you could add the argument parse_dates=[] to avoid
  transforming the dates from string into datetime object later.
* [x] When creating dummy variables, you should not manually type in the names
  of columns. You should write a function to return a list of categorical
  variables.
* [ ] You should discretize float variables like
  "total_price_including_optional_support" and "students_reached" since the
  range of values is not standardized.
* [ ] In the time split function, what you did is incorrect. You only equally
  partition the whole dataset into 4 parts but lost the time information. What
  we want is to seperate the data into 4 6-months windows. You should not drop
  the date_posted column and use the timestamps to split into train/test set.
  Also, you should write a function of dataset instead keeping a list.
* [ ] There is no need to plot the f1 curve for different classifier. The
  result df should be concatenated vertically instead of horizontally. You
  should never use predict(), which has been emphasized many times by Rayid.
  AOC would be a much better metric than f1 with default threshold.
* [ ] For logistic regression, the parameter that you need to change is C
  instead "L1" or "L2".
* [ ] The performance of SVM is suspicious, C should be chosen from [1,10,100]
  so on.
* [ ] In random forest and bagging, the number of trees should be larger. 10 is
  really small and doesn't make sense.
* [ ] You should compare the AOC instead of F1 for different models. You should
  also plot precision vs recall curve.
* [ ] Missing problem analysis, i.e. how would you use your trained model to
  interpret the problem and what suggestion you can give.
* [ ] You should combine precision and recall curves in a single graph. You
  need to select the best parameter for each model in the holdoutset. The
  prediction function based on population threshold could be improved.

# Picking Projects on DonorsChoose.org: Recommendation
## Goal
The website DonorsChoose.org wants to help projects find funding. It has a
landing page that users of the site see every time they go to the site. The
product team would like to feature projects on the landing page that otherwise
might not get funded.

Ideally, we would like to develop a machine learning algorithm that can predict
which projects will not get funded within 60 days. The product team can then
feature only projects from this predicted set on the landing page.

The product team only has room on the landing page to feature 5% of all
projects on DonorsChoose.org. This is our available intervention.

## Metric Choice
Since more than 5% of projects do not get funded within 60 days, then the
metric that is probably most important here is precision. We cannot hope to
feature all projects that need to be featured, so instead we want to focus on
making sure that the projects we feature do in fact need help.

The best we could do would be to make sure that every project we feature would
not otherwise have been funded. That would mean a precision of 100%; in this
world we would reduce the number of projects being unfunded by as much as we
possibly can.

## Model Evaluation
Having trained lots of models on our input data, we need to carefully consider
which model will be the best model according to our target metric.

If you examine Figure 1, you will see that at the 5% threshold the model with
the best precision is the boosting classifier (followed very quickly by the
bagging classifier). At this threshold, both models are able to predict with
roughly 52% precision.

This does not sound like a particularly impressive figure, because it means
that one out of every two times we feature a project, it would have been funded
anyway. But this model performs significantly better than picking randomly,
which would yield a project in need of funding only 29% of the time.

The precision might be higher if we lowered our threshold, but this would mean
that we are reaching fewer projects in need of funding. Figure 2 shows how
recall falls as we lower the threshold, and Figure 3 shows how this fall in
recall offsets the gain in precision.

So we ought to deploy the boosting classifier in production.

## Performance Over Time
The last figure, Figure 4, shows how the precision of several models varies
over time. This graph is showing precision at the 5% threshold.

What this graph tells us is that precision will vary going forward. Just
because we saw precision of 52% when the model was used on our test set does
not mean we will keep seeing that value going forward. In fact, the model did
better on the second time split than on the final test set, showing exactly how
precision could fall (though also rise) for new data.

![Precision](images/precision.png)

![Recall](images/recall.png)

![F1](images/f1.png)

![Precision Over Time](images/precision_splits.png)

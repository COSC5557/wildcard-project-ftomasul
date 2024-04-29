# Finn Tomasula Martin
# COSC-4557
# Wildcard
# This file contains the code for the wildcard exercise

# Clear environment
rm(list = ls())
while (!is.null(dev.list())) dev.off()

# Load in libraries
library(mlr3verse)

# Load in data
wine = read.csv("winequality-red.csv", sep=";")

# Set seed for reproducibility
set.seed(123)

# Define task
task = as_task_classif(wine, target = "quality")

# Define preprocessing PipeOps
scale = po("scale", robust = TRUE)
balance = po("classbalancing", ratio = 6)

# Define models to be optimized
kknn = lrn("classif.kknn", k = to_tune(1, 20), distance = to_tune(1, 20), predict_type = "prob", id = "kknn")
rpart = lrn("classif.rpart", maxdepth = to_tune(1, 30), minbucket = to_tune(1, 50), id = "rpart")
ranger = lrn("classif.ranger", num.trees = to_tune(100, 1000), mtry = to_tune(1, 8), min.node.size = to_tune(1, 10), predict_type = "prob", id = "ranger")

# Define full pipelines to be optimized
kknn_pipe = as_learner(scale %>>% balance %>>% 
                         gunion(list(
                           po("learner_cv", lrn("classif.kknn", predict_type = "prob")),
                           po("learner_cv", lrn("classif.ranger", predict_type = "prob")))) %>>%
                         po("featureunion") %>>% kknn)
rpart_pipe = as_learner(scale %>>% balance %>>% 
                          gunion(list(
                            po("learner_cv", lrn("classif.kknn", predict_type = "prob")),
                            po("learner_cv", lrn("classif.ranger", predict_type = "prob")))) %>>%
                          po("featureunion") %>>% rpart)
ranger_pipe = as_learner(scale %>>% balance %>>% 
                           gunion(list(
                             po("learner_cv", lrn("classif.kknn", predict_type = "prob")),
                             po("learner_cv", lrn("classif.ranger", predict_type = "prob")))) %>>%
                           po("featureunion") %>>% ranger)

# Define optimization process
tuner = tnr("mbo")

kknn_instance = ti(
  task = task,
  learner = kknn_pipe,
  resampling = rsmp("cv", folds = 10),
  measures = msrs(c("classif.ce", "time_train")),
  terminator = trm("evals", n_evals = 50)
)

rpart_instance = ti(
  task = task,
  learner = rpart_pipe,
  resampling = rsmp("cv", folds = 10),
  measures = msrs(c("classif.ce", "time_train")),
  terminator = trm("evals", n_evals = 50)
)

ranger_instance = ti(
  task = task,
  learner = ranger_pipe,
  resampling = rsmp("cv", folds = 10),
  measures = msrs(c("classif.ce", "time_train")),
  terminator = trm("evals", n_evals = 50)
)

kknn_results = tuner$optimize(kknn_instance)
rpart_results = tuner$optimize(rpart_instance)
ranger_results = tuner$optimize(ranger_instance)











# Load libraries
library(tidyverse)
library(tidymodels)
library(lightgbm)
library(recipes)
library(vroom)
library(bonsai)
library(embed)


##EDA##
train_df <- vroom('train.csv')
test_df <- vroom('test.csv')

summary(train_df)
str(train_df)

#Fix column error in preds fn
continuous_cols <- grep("^cont", names(train_df), value = TRUE)

for(col in continuous_cols) {
  if(col %in% names(test_df)) {
    # Ensure both are numeric
    train_df[[col]] <- as.numeric(train_df[[col]])
    test_df[[col]] <- as.numeric(test_df[[col]])
  }
}

train_df$loss <- as.numeric(gsub("[^0-9.-]", "", as.character(train_df$loss)))
train_df$loss[is.na(train_df$loss)] <- median(train_df$loss, na.rm = TRUE)

##Target Encoding
my_recipe <- recipe(loss ~ ., data = train_df) |>
  step_rm(id) |> 
  step_other(all_nominal_predictors(), threshold = .001) |> 
  update_role(loss, new_role = "outcome") |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(loss)) |> 
  step_corr(all_numeric_predictors(), threshold = 0.6) |> 
  step_normalize(all_numeric_predictors())|> 
  step_zv(all_predictors())

##Test Recipe##
#prep <- prep(my_recipe)
#baked <- bake(prep, new_data = train_df)
#str(baked)

##Boosted Tree##
boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")


##Workflow##
acs_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

##Tuning Grid
grid_of_tune <- grid_regular(tree_depth(),
                             trees(),
                             learn_rate(),
                             levels = 3)

##CV Split
set.seed(294)
folds <- vfold_cv(train_df, v = 5)

#Run CV
CV_results <- acs_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tune,
            metrics = metric_set(mae))

bestTune <- CV_results %>%
  select_best(metric = "mae")

final_wf <- acs_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_df)

##Prediction Time##
preds <- predict(final_wf,
                 new_data = test_df)

submission <- data.frame(
  id = test_df$id,
  loss = preds$.pred)

#Remove any duplicates from submission
submission <- submission[!duplicated(submission$id), ]

#Write out the file to submit to Kaggle
vroom_write(x = submission, file = "./boosted2.csv", delim = ",")





# Clear everything - removes all objects from workspace
#rm(list = ls())




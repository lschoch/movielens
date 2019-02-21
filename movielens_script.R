#############################################################
# Create edx set and validation set
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings

# validation <- validation %>% select(-rating)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# RMSE function
#############################################################

# Create a function that computes the residual mean squared error (RMSE) for vectors of
# ratings and their corresponding predictors:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#############################################################
# Effects model
#############################################################
# Partition edx into a training set and a test set
set.seed(2019)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]
# Make sure userId and movieId in test_set are also in train_set
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
# Add rows removed from test_set back into train_set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
# Remove objects no longer needed
rm(temp, test_index, removed)

# Calculate mu, mean rating of the training set, to be used in the prediction formula
mu <- mean(train_set$rating)

# Optimize the regularization tuning parameter, lambda
# Calculate rmses over a range of lambdas and find the lambda that minimizes rmse
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l)) # b_i = movie effect
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l)) # b_u = user effect
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarise(b_g = sum(rating - b_i - b_u - mu)/(n()+l)) # b_g = genre effect
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>% 
    .$pred
  # calculate RMSE  
  return(RMSE(predicted_ratings, test_set$rating))
})

# determine optimum lambda that minimizes the rmses
optimum_lambda <- lambdas[which.min(rmses)]

# Calculate final model using optimum_lambda
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+optimum_lambda))
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+optimum_lambda))
b_g <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>% 
  summarise(b_g = sum(rating - b_i - b_u - mu)/(n()+optimum_lambda))
predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

#############################################################
# Rborist model
#############################################################

# create new object from edx, adding b_i and b_u and b_g as columns
x <- edx %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres")

# add a column for number of ratings per user
temp <- x %>% group_by(userId) %>% summarise(no.usr.ratings = n()) %>% ungroup()
x <- x %>% left_join(temp, by = "userId")
rm(temp)

# add columns for rating year and movie year
x <- x %>% 
  mutate(rating_yr = as.numeric(format(round_date(as_datetime(timestamp), "day"), "%Y")), 
         movie_yr = as.numeric(str_extract(title, "\\d{4}")))

# take a random sample of the data to decrease computation time
N <- 500000 # sample size
set.seed(2019)
x <- x[sample(1:nrow(x), N, replace = FALSE), ]

# create objects to be used for training the model
y <- x$rating
x <- x %>% select(b_i, b_u, b_g, no.usr.ratings, rating_yr, movie_yr)

# calculate Rborist model with preprocessing and parameter tuning
# use ntree = 50 to save computation time during parameter optimization
set.seed(2019)
grid <- expand.grid(predFixed = seq(1, 6), minNode = seq(50, 150, 25))
train_rf <- train(x, y,
                  method = "Rborist",
                  preProcess = "center",
                  trControl=trainControl(method = 'cv', number=5, p = 0.8),
                  nTree = 50,
                  tuneGrid = grid)

#calculate final model with ntree = 500 (default) and bestTune parameters
train_rf_final <- train(x, y,
                        method = "Rborist",
                        preProcess = "center",
                        trControl=trainControl(method = 'cv', number=5, p = 0.8),
                        nTree = 500,
                        tuneGrid = data.frame(predFixed = train_rf$bestTune$predFixed, 
                                              minNode = train_rf$bestTune$minNode))

#############################################################
# RESULTS
#############################################################

#--- use the effects model to predict ratings for the validation set---#

# modify the validation set to include the three effects variables
validation_effects <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") 

# CALCULATE THE PREDICTED RATINGS FOR THE EFFECTS MODEL
validation_effects_predict <- validation_effects %>%  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

# CALCULATE THE RMSE FOR THE EFFECTS MODEL
validation_effects_rmse <- RMSE(validation_effects_predict, validation$rating)

########################################################################

#--- use the rborist model to predict ratings for the validation set---#

# add the three effects variables to the validation dataset
validation_rborist <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres")

# add the column for number of ratings per user
temp <- validation_rborist %>% group_by(userId) %>% summarise(no.usr.ratings = n()) %>% ungroup()
validation_rborist <- validation_rborist %>% left_join(temp, by = "userId")
rm(temp)

# add the columns for rating year and movie year
validation_rborist <- validation_rborist %>% 
  mutate(rating_yr = as.numeric(format(round_date(as_datetime(timestamp), "day"), "%Y")), 
         movie_yr = as.numeric(str_extract(title, "\\d{4}")))

# select the six columns that were used for training
validation_rborist <- validation_rborist %>% select(b_i, b_u, b_g, no.usr.ratings, rating_yr, movie_yr)

# CALCULATE THE PREDICTED RATINGS FOR THE RBORIST MODEL
validation_rborist_predict <- predict(train_rf_final, newdata = validation_rborist)

# CALCULATE THE RMSE FOR THE RBORIST MODEL
validation_rborist_rmse <- RMSE(validation_rborist_predict, validation$rating)

# OUTPUT
sprintf("The RMSE for the effects model is: %s", round(validation_effects_rmse, digits = 5))
sprintf("The RMSE for the Rborist model is: %s", round(validation_rborist_rmse, digits = 5))
sprintf("Validation_effects_predict and validation_Rborist_predict are the vectors of predicted ratings.")


# Project: MovieLens, based on Movielens 10M dataset
# Author : Elena Oskrogo
# Harvard Edx PH125.9x Data Science: Capstone

#install package tinytex & load library to compile Markdown report in pdf format
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")  
library(tinytex)

#install additional packages & load library needed for visualization, analysis, etc.
if(!require(dplyr))      install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate))  install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(tidyr))      install.packages("tidyr)", repos = "http://cran.us.r-project.org")
if(!require(scales))     install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(tidytext))   install.packages("tidytext", repos = "http://cran.us.r-project.org")
if(!require(tidyverse))  install.packages("tidyverse",  repos = "http://cran.us.r-project.org")
if(!require(caret))      install.packages("caret",      repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(dplyr)
library(lubridate)
library(tidyr)
library(scales)
library(tidytext)
library(tidyverse)
library(caret)
library(data.table)

# load data as provided in edx course
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#load data
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId") #original data set

# end of load section

#remove data that will be no longer needed
rm(dl, ratings, movies)

#initial data description
#preview of first 6 rows of movielens
head(movielens)

#dimensions of movielens - original data
predictors <- dim(movielens)[2]
observations <- dim(movielens)[1]

#explore available features
#explore if any predictors has N/A values
movielens %>%  filter(is.na(userId)) %>% summarize(count=n()) # check how many user is N/A - 0 in this dataset
movielens %>%  filter(is.na(movieId)) %>% summarize(count=n()) # check how many movies is N/A - 0 in this dataset
movielens %>%  filter(is.na(rating)) %>% summarize(count=n()) # check how many rating is N/A - 0 in this dataset
movielens %>%  filter(is.na(timestamp)) %>% summarize(count=n()) # check how many timestamp is N/A - 0 in this dataset
movielens %>%  filter(is.na(title)) %>% summarize(count=n()) # check how many titles is N/A - 0 in this dataset
movielens %>%  filter(is.na(genres)) %>% summarize(count=n()) # check how many genres is N/A - 0 in this dataset

#explore if any predictors has null values
movielens %>%  filter(is.null(userId)) %>% summarize(count=n()) # check how many user is NULL - 0 in this dataset
movielens %>%  filter(is.null(movieId)) %>% summarize(count=n()) # check how many movies is NULL - 0 in this dataset
movielens %>%  filter(is.null(rating)) %>% summarize(count=n()) # check how many rating is NULL - 0 in this dataset
movielens %>%  filter(is.null(timestamp)) %>% summarize(count=n()) # check how many timestamp is NULL - 0 in this dataset
movielens %>%  filter(is.null(title)) %>% summarize(count=n()) # check how many titles is NULL - 0 in this dataset
movielens %>%  filter(is.null(genres)) %>% summarize(count=n()) # check how many genres is NULL - 0 in this dataset

#explore if any predictors has not expected type i.e. character instead of numeric, etc
movielens %>%  filter(!is.numeric(userId)) %>% summarize(count=n()) # check how many user have wrong format - 0 in this dataset
movielens %>%  filter(!is.numeric(movieId)) %>% summarize(count=n()) # check how many movies have wrong format - 0 in this dataset
movielens %>%  filter(!is.numeric(rating)) %>% summarize(count=n()) # check how many rating have wrong format - 0 in this dataset
movielens %>%  filter(!is.numeric(timestamp)) %>% summarize(count=n()) # check how many timestamp have wrong format - 0 in this dataset
movielens %>%  filter(!is.character(title)) %>% summarize(count=n()) # check how many titles have wrong format - 0 in this dataset
movielens %>%  filter(!is.character(genres)) %>% summarize(count=n()) # check how many titles have wrong format - 0 in this dataset

#data transformation
#convert field timestamp to date format
movielens_transformed <- movielens %>% mutate(date = as_datetime(timestamp))
#extract year when movie was produced
movielens_transformed <- movielens_transformed %>% mutate(YearM=as.integer(str_extract(str_extract(title," \\([0-9]{4}\\)$"),"[0-9]{4}")))
#remove movie production year from title
movielens_transformed <- movielens_transformed %>% mutate(Title=str_remove(title," \\([0-9]{4}\\)$"))
# calculate movie age at the moment of rating
movielens_transformed <- movielens_transformed %>% mutate(Age=year(date)-YearM)
movielens_transformed <- movielens_transformed %>% select(userId, movieId, Title, YearM, genres, rating, date, Age)
# for future model, calculate how many words contain title
edx_movie <- movielens_transformed %>% select(movieId, Title) %>% distinct(movieId, Title)  %>% unnest_tokens(word, Title) %>% 
  group_by(movieId) %>% summarize(Word_title = n())
movielens_transformed <-  left_join(movielens_transformed, edx_movie, by="movieId")

# verify if any of newly created fields has N/A value, all counts where 0
movielens_transformed %>%  filter(is.na(Word_title)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.na(date)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.na(YearM)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.na(Age)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.na(Title)) %>% summarize(count=n())

# verify if any of newly created fields has null value, all counts where 0
movielens_transformed %>%  filter(is.null(Word_title)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.null(date)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.null(YearM)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.null(Age)) %>% summarize(count=n())
movielens_transformed %>%  filter(is.null(Title)) %>% summarize(count=n())

# check if all columns values are in expected format

is.POSIXct <- function(x) inherits(x,"POSIXct") # create function to check data format

movielens_transformed %>%  filter(!is.numeric(Word_title)) %>% summarize(count=n())
movielens_transformed %>%  filter(!is.POSIXct(date)) %>% summarize(count=n())
movielens_transformed %>%  filter(!is.numeric(YearM)) %>% summarize(count=n())
movielens_transformed %>%  filter(!is.numeric(Age)) %>% summarize(count=n())
movielens_transformed %>%  filter(!is.character(Title)) %>% summarize(count=n())

#overview of transformed data set
head(movielens_transformed)
summary(movielens_transformed)

#explore in details each predictors
#explore rating
gg_rating_histo <- movielens_transformed %>% ggplot(aes(rating))+ geom_histogram() + 
  ggtitle("Rating histogram") + xlab("Rate") + ylab("Count")

#UserId exploration
#how many different users
nbr_users <- movielens_transformed %>% select(userId) %>% summarize(user=unique(userId)) %>% summarize(count=n()) #Total of 69878 unique users
#How often they rate movie
min_nbr_rates <- movielens_transformed %>% select(userId) %>% group_by(userId) %>% summarize(count=n()) %>% summarize(min=min(count))
max_nbr_rates  <- movielens_transformed %>% select(userId) %>% group_by(userId) %>% summarize(count=n()) %>% summarize(max=max(count))
#visualization of frequency of rating per user
gg_nbr_rates <- movielens_transformed %>% group_by(userId) %>% summarize(nbr_rates=n()) %>% ggplot(aes(nbr_rates)) + geom_histogram(bins=30, color = "black") +
             ggtitle("Histogram rating frequency per user") + xlab("Number of rating") + ylab("Users")
# sample -  number of users provided less than 100 rates and less than 400
user_100q <- movielens_transformed %>% group_by(userId) %>% summarize(nbr_rates=n()) %>% filter(nbr_rates < 100) %>% summarize(total=n())
user_400q <- movielens_transformed %>% group_by(userId) %>% summarize(nbr_rates=n()) %>% filter(nbr_rates < 400) %>% summarize(total=n())

# explore how long users provide rating
# build data frame user_observation with UserId, total number of rating, date of first feedback amd date of latest feedback
user_nbr_rate <- movielens_transformed %>% select(userId) %>% group_by(userId) %>% summarize(count=n()) 
user_start_rate <- movielens_transformed %>% select(userId, date) %>% group_by(userId) %>% summarize(start=min(date))
user_end_date <- movielens_transformed %>% select(userId, date) %>% group_by(userId) %>% summarize(last=max(date))
user_observation <- left_join(user_nbr_rate, user_start_rate, by="userId")
user_observation <- left_join(user_observation, user_end_date, by="userId")
user_observation <- user_observation %>% mutate(duration = difftime(last, start, units = "days") + 1, rate_per_day = count / as.numeric(duration) )

# how long user provide feedback
user_obs_long <- user_observation %>% summarize(min = min(duration), max = max(duration))

# build table with group of users provided x feedback per day
nbr <- user_observation %>% filter(rate_per_day < 1) %>% summarize(nbr = n())
user_feedback <- data.frame(From = '0', To = '1', Users = nbr$nbr )
nbr <- user_observation %>% filter(rate_per_day > 1 & rate_per_day <= 2) %>% summarize(nbr = n())
user_feedback <- bind_rows(user_feedback, data.frame(From = '1', To = '2', Users = nbr$nbr ))
nbr <- user_observation %>% filter(rate_per_day > 2 & rate_per_day <= 4) %>% summarize(nbr = n())
user_feedback <- bind_rows(user_feedback, data.frame(From = '2', To = '4', Users = nbr$nbr ))
nbr <- user_observation %>% filter(rate_per_day > 4 & rate_per_day <= 6) %>% summarize(nbr = n())
user_feedback <- bind_rows(user_feedback, data.frame(From = '4', To = '6', Users = nbr$nbr ))
nbr <- user_observation %>% filter(rate_per_day > 6 & rate_per_day <= 12) %>% summarize(nbr = n())
user_feedback <- bind_rows(user_feedback, data.frame(From = '6', To = '12', Users = nbr$nbr ))
nbr <- user_observation %>% filter(rate_per_day > 12) %>% summarize(nbr = n())
user_feedback <- bind_rows(user_feedback, data.frame(From = '12', To = 'Max' , Users = nbr$nbr ))

# build table with rate_per_date - number of users
tab_user_feedback <- user_feedback %>% knitr::kable()

# relationship between period of observation and number of feedback
gg_user_observation <- user_observation %>% ggplot(aes(x = rate_per_day, y = as.numeric(duration))) + geom_point() +
  ggtitle("Rates per days versus activity period") + xlab("Number of rates per day") + ylab("Rating period")

#explore average rating per user
gg_user_rating <- movielens_transformed %>% group_by(userId) %>% summarize(user_rating = mean(rating)) %>% ggplot(aes(user_rating)) + geom_histogram(bins=30, color = "black") + 
  ggtitle("Rating per user") + xlab("Average Rate") + ylab("Count")

user_3r <- movielens_transformed %>% group_by(userId) %>% summarize(user_rating = mean(rating)) %>% filter(user_rating <= 3) %>% count(total=n())
user_rat <- data_frame(Rating = "<= 3", Users=user_3r$total, Percentage= user_3r$total/nbr_users$count*100)
user_34r <- movielens_transformed %>% group_by(userId) %>% summarize(user_rating = mean(rating)) %>% filter(user_rating > 3 & user_rating <= 4) %>% count(total=n())
user_rat <- bind_rows(user_rat, data_frame(Rating = "> 3 & <= 4", Users=user_34r$total, Percentage= user_34r$total/nbr_users$count*100))
user_4r <- movielens_transformed %>% group_by(userId) %>% summarize(user_rating = mean(rating)) %>% filter(user_rating > 4) %>% count(total=n())
user_rat <- bind_rows(user_rat, data_frame(Rating = "> 4", Users=user_4r$total, Percentage= user_4r$total/nbr_users$count*100))
user_tab <- user_rat %>% knitr::kable()

#explore how many diff movies
nbr_movies <- movielens_transformed %>% select(movieId) %>%  summarize(movie=unique(movieId)) %>% summarize(count=n())
#how often movie are rated
min_nbr_m_rates <- movielens_transformed %>% select(movieId) %>% group_by(movieId) %>% summarize(count=n()) %>% summarize(min=min(count))
max_nbr_m_rates  <- movielens_transformed %>% select(movieId) %>% group_by(movieId) %>% summarize(count=n()) %>% summarize(max=max(count))
gg_nbr_movie_rating <- movielens_transformed %>% group_by(movieId) %>% summarize(nbr_rates=n())  %>% ggplot(aes(nbr_rates)) + geom_histogram(bins=30, color = "black") +
                    ggtitle("Histogram rating frequency per movie") + xlab("Number of rating") + ylab("Movies")

# explore average rating per movie
gg_movie_rating <- movielens_transformed %>% group_by(movieId) %>% summarize(movie_rating = mean(rating)) %>% ggplot(aes(movie_rating)) + geom_histogram(bins=30, color = "black") +
  ggtitle("Movie rating histogram") + xlab("Average movie Rate") + ylab("Count")

#explore relationship users - movie
#heatmap for user-movie selection
gg_heatmap <- movielens_transformed %>% select(userId,movieId, rating) %>%  ggplot(aes(x=userId, y= movieId)) + geom_tile(aes(fill=rating)) +
                        ggtitle("Heatmap User - Movie") + xlab("User") + ylab("Movie")
#heatmap for user-movie selection (partial)
set.seed(1970, sample.kind="Rounding")
selection_user <- sample(1:nbr_users$count,10000,replace=FALSE )
selection_movie  <- sample(1:nbr_movies$count,10000,replace=FALSE )
 
movielens_transformed %>% select(userId,movieId, rating) %>% filter((userId %in% selection_user) & (movieId %in% selection_movie)) %>% ggplot(aes(x=userId, y= movieId)) + geom_tile(aes(fill=rating)) +
  ggtitle("Heatmap User - Movie (partial view for 10000 users/ movies)") + xlab("User") + ylab("Movie")

# explore YearM - movie release year
gg_year <- movielens_transformed %>% select(movieId, YearM) %>% distinct(movieId, YearM) %>% group_by(YearM) %>% summarize(count=n()) 

gg_year_p <- gg_year %>% ggplot(aes(x=YearM, y=count)) + geom_line()  + 
             ggtitle("Movie production") + xlab("Year") + ylab("Movies")

# explore genres
# how many different genres we have
genres <- movielens_transformed %>% select(genres) %>% distinct(genres)
nbr_genres <- dim(genres)[1]
# how many movies per genres
genres_m <- movielens_transformed %>% select(movieId, genres) %>% distinct(movieId, genres) %>% group_by(genres) %>% summarize(count=n()) 
genres_m_top10 <- head(arrange(genres_m, -count),10)
genres_m_bot10 <- head(arrange(genres_m, count),10)
# frequency of rating per genre
genres_temp <- movielens_transformed %>% group_by(genres) %>% summarize(count=n()) 
min_nbr_genres <- genres_temp %>% summarize(min = min(count))
max_nbr_genres <- genres_temp %>% summarize(max = max(count))
gg_genres <- genres_temp %>%  mutate(genres = reorder(genres,count)) %>% 
             ggplot(aes(x=genres,y=count)) + coord_flip() + geom_point() +
             ggtitle("Nbr of ratings per genre") + xlab("Genres") + ylab("Nbr of ratings") + 
             theme(axis.line = element_blank(), axis.text.x = element_blank(), axis.text.y = element_blank())
# top 10 genres
genres_top <- head(arrange(genres_temp, -count),10) 
# bottom 10 genres
genres_bot <- head(arrange(genres_temp, count),10)

# How rating value influenced by genre
genres_temp_r <- movielens_transformed %>% group_by(genres) %>% summarize(average_rate = mean(rating)) 
gg_genres_r <- genres_temp_r %>%  mutate(genres = reorder(genres,average_rate)) %>% 
  ggplot(aes(x=genres,y=average_rate)) + coord_flip() + geom_point() +
  ggtitle("Average rate per genre") + xlab("Genres") + ylab("Average rate") + 
  theme(axis.line = element_blank(), axis.text.x = element_blank(), axis.text.y = element_blank())
# top 10 genres rates
genres_top_r <- head(arrange(genres_temp_r, -average_rate),10) 
# bottom 10 genres rates
genres_bot_r <- head(arrange(genres_temp_r, average_rate),10)

# explore Age column
gg_age <- movielens_transformed %>% group_by(Age) %>% summarize(count = n()) %>% ggplot(aes(x = Age, y = count)) + geom_line() +
  ggtitle("Nbr ratings - movie age") + xlab("Movie Age") + ylab("Nbr ratings")
#Records with negative movie age - rating was provided before movie distribution
yearly_reviewed <- movielens_transformed %>% filter(Age < 0) %>% summarize(count=n())

gg_age_rate <- movielens_transformed %>% group_by(Age) %>% summarize(Mean = mean(rating)) %>% ggplot(aes(x = Age, y = Mean)) + geom_line()  +
  ggtitle("Average rating - movie age") + xlab("Movie Age") + ylab("Ratings")

# Impact of Word_title on rating
gg_word_rate <- movielens_transformed %>% group_by(Word_title) %>% summarize(Mean = mean(rating)) %>% ggplot(aes(x = Word_title, y = Mean)) + geom_line()  +
  ggtitle("Average rating - Word's in Movie title") + xlab("Word's in Movoe Title") + ylab("Ratings")

# explore date of rating column
# how many ratings per year
date_rating <- movielens_transformed %>% mutate(year = year(date)) %>% group_by(year) %>% summarize(count=n()) 
gg_date_rating <- date_rating %>% ggplot(aes(x = year, y = count)) + geom_line() +
  ggtitle("Nbr ratings per year") + xlab("Year (rating date)") + ylab("Nbr of ratings")

date_rating_y2000 <- movielens_transformed %>% filter(year(date) == 2000) %>%
  mutate(month = month(date)) %>% group_by(month) %>% summarize(count=n()) 
gg_date_rating_y2000 <- date_rating_y2000 %>% ggplot(aes(x = month, y = count)) + geom_line() +
  ggtitle("Ratings in 2000") + xlab("2000 (Months)") + ylab("Nbr of ratings")

date_rating_y2005 <- movielens_transformed %>% filter(year(date) == 2005) %>%
                 mutate(month = month(date)) %>% group_by(month) %>% summarize(count=n()) 
gg_date_rating_y2005 <- date_rating_y2005 %>% ggplot(aes(x = month, y = count)) + geom_line() +
  ggtitle("Ratings in 2005") + xlab("2005 (Months)") + ylab("Nbr of ratings")



# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens_transformed$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens_transformed[-test_index,]
temp <- movielens_transformed[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# remove what no longer needed
rm( test_index, temp, removed)

# end of split section edx - training dataset & validation - validation one

#define RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# build model
#build model as just average 

mu_hat <- mean(edx$rating)
mu_hat

naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

#add movie effect
mu <- mean(edx$rating) 
movie_avgs <- edx %>% filter(!is.na(rating)) %>%
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

gg_movie_impact <- movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black")) +
                   ggtitle("Individual Movie impact") + xlab("Movie impact")

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, validation$rating)
model_1_rmse
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

rmse_results %>% knitr::kable()



#user effect

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

gg_user_impact <- user_avgs %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black")) +
               ggtitle("Add User impact") + xlab("User impact")

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results<- bind_rows(rmse_results,
                         data_frame(method="Movie User Effect Model",
                                    RMSE = model_2_rmse ))

rmse_results %>% knitr::kable()

#add genres effect
genres_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

gg_genres_impact <- genres_avgs %>% qplot(b_g, geom ="histogram", bins = 10, data = ., color = I("black")) +
                    ggtitle("Add Genres impact") + xlab("Genres impact")

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie User Genres Effect Model",
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

#add movie age effect
age_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  group_by(Age) %>%
  summarize(b_a = mean(rating - mu - b_i - b_u - b_g))

gg_age_impact <- age_avgs %>% qplot(b_a, geom ="histogram", bins = 10, data = ., color = I("black")) +
                  ggtitle("Add Age impact") + xlab("Age impact")

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(age_avgs, by='Age') %>%
  mutate(pred = mu + b_i + b_u + b_g+ b_a) %>%
  pull(pred)
model_4_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie User Genres Age Effect Model",
                                     RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()

# Add title short or long impact

count_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(age_avgs, by='Age') %>%  
  group_by(Word_title) %>%
  summarize(b_c = mean(rating - mu - b_i - b_u - b_g - b_a))

gg_count_impact <- count_avgs %>% qplot(b_c, geom ="histogram", bins = 10, data = ., color = I("black")) +
                   ggtitle("Add Title impact") + xlab("Title impact")

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(age_avgs, by='Age') %>%
  left_join(count_avgs, by='Word_title') %>%
  mutate(pred = mu + b_i + b_u + b_g+ b_a+b_c) %>%
  pull(pred)
model_5_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie User Genres Age Title Effect Model",
                                     RMSE = model_5_rmse ))

rmse_res <- rmse_results %>% knitr::kable()


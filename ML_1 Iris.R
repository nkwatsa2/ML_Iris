# -----------------ML project 1------------------#



# Using caret package
library(data.table)

library(dplyr)
library(magrittr)
data("iris") #attaching the dataset iris in the environment

iris_data <- iris #renaming the dataset
class(iris_data)
head(iris_data)
cor(iris[, 1:4])    
# overall correlation btn petal widtha nd petal length
cor(iris_data$Petal.Length, iris_data$Petal.Width) # correlation is pretty high i.e 0.96

#from correlation plot, high correlation btn petal width and length for all species.
# checking individual correlation for the diff species sepal lenth and width, petal lemgth and width

iris_levels <- levels(iris_data$Species) # 3 levels

cor(iris_data[iris_data$Species == iris_levels[1], 1:4]) #setosa
cor(iris_data[iris_data$Species == iris_levels[2], 1:4]) # versicolor
cor(iris_data[iris_data$Species == iris_levels[3], 1:4]) #virginica

# proportion of each species as a percentage
round(prop.table(table(iris_data$Species)) * 100, digits = 5) # digits in this function is for decimal points
# we can see that all species have equal proportions of 33.3333%
iris_data$Species = as.factor(iris_data$Species)
summary(iris_data)
# we could have easily seen from thsi summary that proportion of species is equal

library(caret)
set.seed(1245)
iris_data_indices <- createDataPartition(y = iris_data$Species, p = 0.8, list = FALSE, times = 1)
test_iris_set <- iris_data[-iris_data_indices,]
training_iris_set <- iris_data[iris_data_indices,]

# setting levels for train and test data
levels(training_iris_set$Species) <- make.names(levels(factor(training_iris_set$Species)))
levels(test_iris_set$Species) <- make.names(levels(factor(test_iris_set$Species)))

training_control_iris <- trainControl(method = "cv", 
                                      number = 10, 
                                      repeats = 5)
metric = "Accuracy"

iris_data_knn <- train(Species~. , data = training_iris_set,
                       method = "lda", 
                       preProcess = c("center", "scale"),
                       metric = metric,
                       tr_control = training_control_iris)

set.seed(1245)
fit.knn <- train(Species~., data = training_iris_set, 
                 method = "knn", 
                 metric = metric, 
                 trControl = training_control_iris)

fit.knn
predictions_iris_knn <- predict(fit.knn, newdata = test_iris_set)
predictions_iris_knn
test_iris_set$Species
table(predictions_iris_knn)

# confusion matrix
confusionMatrix(predictions_iris_knn, test_iris_set[, 5])
#warnings()
ggpairs(iris_data)
class(training_iris_set$Species)



#-----------------------------------------------------------------------
library(caTools)
set.seed(103)
splitset <- sample.split(iris_data, SplitRatio = 0.8)
train_iris <- subset(iris_data, splitset == TRUE)
test_iris <- subset(iris_data, splitset == FALSE)

# scaling after removing column containing labels
training_iris <- scale(train_iris[-5])
testing_iris <- scale(test_iris[-5])

library(class)
predictions_iris_knn_2 <- knn(train = training_iris, test = testing_iris,
                              # adding factors for classification
                              cl = train_iris[,5],
                              # Setting 'k' to 5 as it generally avoids overfitting
                              k = 5, prob = T)

predictions_iris_knn_2
predictions_iris_knn

confusionMatrix(predictions_iris_knn_2, test_iris[, 5])

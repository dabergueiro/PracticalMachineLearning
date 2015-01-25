#The goal of your project is to predict the manner in which they did the exercise. 
#This is the "classe" variable in the training set. You may use any of the other variables to predict with. 
#You should create a report describing how you built your model, how you used cross validation, what you think 
#the expected out of sample error is, and why you made the choices you did. You will also use your prediction model 
#to predict 20 different test cases. 
#
#1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing 
#your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5.
#It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed 
#online (and you always want to make it easy on graders :-).
#2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. 
#Please submit your predictions in appropriate format to the programming assignment for automated grading. See the 
#programming assignment for additional details. 

setwd("G:/Coursera/Data Science Specialization/08) Practical Machine Learning/Project")
setwd("M:/Coursera/Data Science Specialization/08) Practical Machine Learning/Project")

#load librarys and test & training datasets
library(caret)
library(rattle)
training <- read.csv("pml-training.csv", header=T)
test <- read.csv("pml-testing.csv", header=T)

#Set seed for reproducibility purposes
set.seed(3262)

#Reduce dataset: remove columns with variance = NA
columns <- colnames(training)
colvars <- numeric(length(columns))

for (i in (1:length(colvars)))
{
  colvars[i] <- var(training[,i])
}

reduced.training <- training[!(is.na(colvars))] #89 vars
reduced.training <- reduced.training[,-c(1,3:7)]

columns <- colnames(reduced.training)
col.numeric <- logical(length(columns))

for (i in (1:length(col.numeric)))
{
  col.numeric[i] <- is.numeric(reduced.training[,i])
}

reduced.training <- reduced.training[col.numeric] #53 variables

reduced.training <- cbind(training$classe, training$user_name, reduced.training)

colnames(reduced.training)[1] <- "classe"
colnames(reduced.training)[2] <- "user_name"


#break training into 2 groups: train and validate
inTrain <- createDataPartition(y=reduced.training$classe, p=0.6, list=F)
train <- reduced.training[inTrain,]
validate <- reduced.training[-inTrain,]

#Define cross-validation
ctrl <- trainControl(method = "cv", number = 10) #using plain k-fold cv due to the size of the training dataset

#First Model: Decision Trees
modelFit <- train(classe~., method="rpart", data=train, trControl=ctrl) #13 minutes to run 4:55
fancyRpartPlot(modelFit$finalModel)

#Confusion Matrix for first model
predictions <- predict(modelFit, newdata=validate)
confusionMatrix(predictions, validate$classe)

#Second Model: Random Forest
modelFit <- train(classe~., method="rf", data=train, trControl=ctrl)

#Confusion Matrix for second model
predictions <- predict(modelFit, newdata=validate)
confusionMatrix(predictions, validate$classe)

#predict on test dataset
predictions <- predict(modelFit, newdata=test)

#output predictions
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)






#use pca to reduce dimensions
preProc <- preProcess(log10(reduced.training[,-c(1,2)]+1), method="pca", pcaComp=2)
trainPC <- predict(preProc, log10(reduced.training[,-c(1:2)]+1))
modelFit <- train(train$classe ~., method="glm", data=trainPC)

#predict using reduced dataset
testPC <- predict(preProc, log10(validate[,()]+1))
confusionMatrix(validate$class, predict(modelFit,testPC))


# import modules --------
library(caret)
library(glmnet)
library(elasticnet)
library(pracma)

train <- read.csv('train.csv')
test <- read.csv('test.csv')

# plot the training data
plot(train$x, train$y)


# piecewise continuous function

fit <- function(x){
  x*(x<=2) + 3**x*(x>2)*(x<4) - x**2*(x>=4)
}

curve(fit(x), -10, 10)
points(2, fit(2))
points(4, fit(4))


# caret package to perform gridsearch


# (d + 1)*(q + 1) - q*d = q + d + 1, number of params for each point
# feature matrix n x (d + 1 + q)
# if d = 2, B0, B1, B2 (fit)
# q number of knots (hyperparam)
# grid search on number of knots (equispaziati)


# knots
# f1(k1) = f2(k1)
# plus continuity in 1st and second derivatives (if cubic)


# grid search between qmin and qmax

# suppose q (number of knots) equals 3
q <- 3
knots <- seq(1/(q+1), 1 - 1/(q+1), 1/(q+1)) # position of knots
plot(train$x, train$y)
points(0.25, mean(train$y[round(train$x, 2) == 0.25]), col = "orchid", cex = 1.5, pch = 19)
points(0.50, mean(train$y[round(train$x, 2) == 0.50]), col = "orchid", cex = 1.5, pch = 19)
points(0.75, mean(train$y[round(train$x, 2) == 0.75]), col = "orchid", cex = 1.5, pch = 19)

x1 <- train$x[train$x < knots[1]]
x2 <- train$x[(train$x <= knots[2]) & (train$x > knots[1])]
x3 <- train$x[(train$x <= knots[3]) & (train$x > knots[2])]
x4 <- train$x[train$x > knots[3]]

y1 <- train$y[train$x < knots[1]]
y2 <- train$y[(train$x <= knots[2]) & (train$x > knots[1])]
y3 <- train$y[(train$x <= knots[3]) & (train$x > knots[2])]
y4 <- train$y[train$x > knots[3]]

# fit 4 functions on the 4 areas (points) separately
# then impose continuity conditions at border

par(mfrow = c(2, 2))
plot(x1, y1, main = "Fun1")
plot(x2, y2, main = "Fun2")
plot(x3, y3, main = "Fun3")
plot(x4, y4, main = "Fun4")

# Function 1
par(mfrow = c(1, 1))
plot(x1, y1, main = "Fun1")
fun1 <- lm(y1~poly(x1, degree=3))
# abline(fun1, col = "blue", lwd = 2)
sd(c(fun1$residuals))
y1_pred <- predict(fun1)
ix1 <- sort(x1, index.return=T)$ix
lines(x1[ix1], y1_pred[ix1], col='blue', lwd=2)

fun1$coefficients[[1]]

par(mfrow = c(1, 1))
prova <- function(x){
  beta0 = fun1$coefficients[[1]]
  beta1 = fun1$coefficients[[2]]
  beta2 = fun1$coefficients[[3]]
  beta3 = fun1$coefficients[[4]]
  out <- beta1*x + beta2*(x**2) + beta3*(x**3)
  return(out)
}

plot(x1, y1, main = "Fun1")
curve(prova(x), 0, 0.25, add = T)
lines(x1[ix1], y1_pred[ix1], col='blue', lwd=2)
summary(fun1)

# Function 2
plot(x2, y2, main = "Fun2")
fun2 <- lm(y2~poly(x2, degree=3))
summary(fun2)
# abline(fun2, col = "blue", lwd = 2)
sd(c(fun2$residuals))
y2_pred <- predict(fun2)
ix2 <- sort(x2, index.return=T)$ix
lines(x2[ix2], y2_pred[ix2], col='blue', lwd=2)

# Function 3
plot(x3, y3, main = "Fun3")
fun3 <- lm(y3~poly(x3, degree=3))
# fun3 <- lm(y3~x3)
# abline(fun3, col = "blue", lwd = 2)
sd(c(fun3$residuals))
y3_pred <- predict(fun3)
ix3 <- sort(x3, index.return=T)$ix
lines(x3[ix3], y3_pred[ix3], col='blue', lwd=2)

# Function 4
plot(x4, y4, main = "Fun4")
# fun4 <- lm(y4~x4)
fun4 <- lm(y4~poly(x4, 3))
# abline(fun4, col = "blue", lwd = 2)
sd(c(fun4$residuals))
y4_pred <- predict(fun4)
ix4 <- sort(x4, index.return=T)$ix
lines(x4[ix4], y4_pred[ix4], col='blue', lwd=2)

# all together
plot(train$x, train$y)
lines(x1[ix1], y1_pred[ix1], col='blue', lwd=2)
lines(x2[ix2], y2_pred[ix2], col='red', lwd=2)
lines(x3[ix3], y3_pred[ix3], col='green', lwd=2)
lines(x4[ix4], y4_pred[ix4], col='orange', lwd=2)
abline(v=0.25)
abline(v=0.50)
abline(v=0.75)



par(mfrow = c(1,1))
plot(train$x, train$y)
abline(fun1, col = "blue", lwd = 2)
x <- train$x
ix <- sort(x, index.return=T)$ix
lines(x[ix], y4_pred[ix], col='blue', lwd=2)

# plot
par(mfrow = c(2, 2))
plot(x1, y1, main = "Fun1")
lines(x1[ix1], y1_pred[ix1], col='blue', lwd=2)
plot(x2, y2, main = "Fun2")
lines(x2[ix2], y2_pred[ix2], col='blue', lwd=2)
plot(x3, y3, main = "Fun3")
lines(x3[ix3], y3_pred[ix3], col='blue', lwd=2)
plot(x4, y4, main = "Fun4")
lines(x4[ix4], y4_pred[ix4], col='blue', lwd=2)

sd(c(c(fun1$residuals), c(fun2$residuals), c(fun3$residuals), c(fun4$residuals)))

# predict(fun1, newdata)

# what i found online -----

sk<-function(x,xi_k){
  xi_1<-0.25 #first knot
  xi_K_1<- 0.5 #second to the last knot
  xi_K<- 0.75   #last knot
  dk<-((x-xi_k)^3*(x>=xi_k)-(x-xi_K)^3*(x>=xi_K))/(xi_K-xi_k)
  dK_1<-((x-xi_K_1)^3*(x>=xi_K_1)-(x-xi_K)^3*(x>=xi_K))/(xi_K-xi_K_1)
  sk<-(xi_K-xi_k)*(dk-dK_1)/(xi_K-xi_1)^2 # scaled sk
  return(sk)
}


s1<-sk(train$x, 0.25)
s2<-sk(train$x, 0.5)
train_new<-cbind(train,s1,s2)
head(train_new,100)

spline_by_hand<-lm(y~x+s1+s2,data=train)
summary(spline_by_hand)

plot(spline_by_hand)
sd(c(spline_by_hand$residuals))

# to do -------------------

# try with k = 3
# then automate using grid search on k and d
# then try for also non-equidistanced knots

# example ------------

d = 2
q = 3
knots = c(0.4, 0.6, 0.8)
n = nrow(train)

X <- matrix(0, n, d + 1 + q)
X[,1] <- 1 # intercept
X[,2] <- train$x
X[,3] <- (train$x)**2
X[,4] <- pmax(0, train$x - rep(knots[1], n))**d
X[,5] <- pmax(0, train$x - rep(knots[2], n))**d
X[,6] <- pmax(0, train$x - rep(knots[3], n))**d

fun <- lm(train$y ~ X)
sd(fun$residuals) # 3033.68

plot(train$x, train$y)
ypred <- predict(fun)
ix <- sort(train$x, index.return=T)$ix
lines(train$x[ix], ypred[ix], col='blue', lwd=2)

# automated version equispaced -----------

# training set error -------------

d = 3
n = nrow(train)
qmin = 1
qmax = 3
errors <- c(rep(NA, qmax))
for(q in qmin:qmax){
  X <- matrix(0, n, d+q)
  knots = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1))
  for(i in 1:d){
    X[,i] <- (train$x)**i
  }
  for(j in 1:q){
    idx = d+j
    X[,idx] <- pmax(0, train$x - rep(knots[j], n))**d
  }
  model <- lm(train$y ~ X)
  errors[q] <- sd(model$residuals)
}

errors

# val set error ----------

# for each value of q: cv
# train the model (training data)
# validation
# test


# FUNCTION

bestmodel <- function(qmax=1, dmax=1, train_x, train_y){
  D <- 1:dmax
  Q <- 1:qmax
  combinations <- data.frame(expand.grid(D, Q))
  rmse <- c(rep(NA, nrow(combinations)))
  
  for(row in 1:nrow(combinations)){
    d = combinations[row, 1]
    q = combinations[row, 2]
    X <- matrix(0, n, d+q)
    knots = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1))
    for(i in 1:d){
      X[,i] <- (train_x)**i
    }
    for(j in 1:q){
      idx = d+j
      X[,idx] <- pmax(0, train_x - rep(knots[j], n))**d
    }
    train.control <- trainControl(method = "cv", 
                                  number = 10)
    # Train the model
    data <- data.frame(cbind(train_y, X))
    colnames(data)[1] <- "y"
    model <- train(y ~., data = data, method = "lm",
                   trControl = train.control)
    rmse[row] <- model[["results"]][["RMSE"]]
  }
  idx_min <- combinations[which.min(rmse),]
  return(list(idx_min, X, rmse))
}

set.seed(123)
idx_min <- bestmodel(50, 3, train$x, train$y)


# best model

d = 1
q = 9
n = nrow(train)
X <- matrix(0, n, d+q)
knots = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1))
for(i in 1:d){
  X[,i] <- (train$x)**i
}
for(j in 1:q){
  idx = d+j
  X[,idx] <- pmax(0, train$x - rep(knots[j], n))**d
}


ntest = nrow(test)
Xtest <- matrix(0, ntest, d+q)
knots = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1))
for(i in 1:d){
  Xtest[,i] <- (test$x)**i
}
for(j in 1:q){
  idx = d+j
  Xtest[,idx] <- pmax(0, test$x - rep(knots[j], ntest))**d
}


train_data <- data.frame(cbind(X, "y" = train$y))
test_data <- data.frame(Xtest)
mod <- lm(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, data = train_data)
sd(mod$residuals)
y_pred <- predict(mod, newdata = test_data)
y <- data.frame("target" = y_pred)
df <- data.frame(cbind("id"=test$id, y))

# prova <- train_data[,-ncol(train_data)]

plot(train$x, train$y)
points(test$x, y_pred, col = "red")

write.csv(df, 'submission1.csv', row.names = F)

## plot -----------

d = idx_min[[1]]
q = idx_min[[2]]
X <- matrix(0, n, d+q)
knots = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1))
for(i in 1:d){
  X[,i] <- (train$x)**i
}
for(j in 1:q){
  idx = d+j
  X[,idx] <- pmax(0, train$x - rep(knots[j], n))**d
}
train.control <- trainControl(method = "cv", 
                              number = 10)
# Train the model
data <- data.frame(cbind(train$y, X))
colnames(data)[1] <- "y"
model <- train(y ~., data = data, method = "lm",
               trControl = train.control)

plot(train$x, train$y)
ypred <- predict(model)
ix <- sort(train$x, index.return=T)$ix
lines(train$x[ix], ypred[ix], col='red', lwd=2)

# Regularization ----------

library(glmnet)

# function
bestmodel_reg <- function(qmax=1, dmax=1, train_x, train_y, alpha=0){
  D <- 1:dmax
  Q <- 1:qmax
  combinations <- data.frame(expand.grid(D, Q))
  rmse <- c(rep(NA, nrow(combinations)))
  for(row in 1:nrow(combinations)){
    d = combinations[row, 1]
    q = combinations[row, 2]
    X <- matrix(0, n, d+q)
    knots = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1))
    for(i in 1:d){
      X[,i] <- (train_x)**i
    }
    for(j in 1:q){
      idx = d+j
      X[,idx] <- pmax(0, train_x - rep(knots[j], n))**d
    }
    cv.out = cv.glmnet(X, y, alpha = alpha, nfolds = 10) 
    bestlam = cv.out$lambda.min
    mod = glmnet(X, y, alpha = 0, lambda = bestlam)
    y_pred = predict(mod, newx = X)
    
    rmse[row] <- sqrt(mean((train_y - y_pred)^2))
  }
  idx_min <- combinations[which.min(rmse),]
  return(list(idx_min, X, rmse))
}

## ridge regression --------------
x <- train$x
y <- train$y

set.seed(123)
bestmodel_reg(50, 3, x, y, 0)

## lasso regression --------------
res = bestmodel_reg(10, 5, x, y, 1)

## elastic net (una via di mezzo tra ridge e lasso regression) ---------------
bestmodel_reg(10, 3, x, y, 0.5)

# tutti uguali, ma il modello fa piu schifo hahah


# NOT equispaced + Regularization -----------

remap <- function(d, q, knots, x){
  n = length(x)
  X <- matrix(0, n, d+q)
  knots = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1))
  for(i in 1:d){
    X[,i] <- (x)**i
  }
  for(j in 1:q){
    idx = d+j
    X[,idx] <- pmax(0, x - rep(knots[j], n))**d
  }
  return(X)
}

# pos can assume values equi (default), quantile, maxcurve
bestmodel_pos <- function(qmax=1, dmax=1, train_x, train_y, alpha=0, pos = 'equi'){
  D <- 1:dmax
  Q <- 1:qmax
  combinations <- data.frame(expand.grid(D, Q))
  rmse <- c(rep(NA, nrow(combinations)))
  features <- list()
  if(pos == 'equi'){
    knots_pos = lapply(Q, function(q)seq(1/(q+1), 1 - 1/(q+1), 1/(q+1)))
  }
  if(pos == 'quantile'){
    knots_pos = lapply(Q, function(q) unname(quantile(train_x,
                                                      probs = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1)))))
  }
  for(row in 1:nrow(combinations)){
    d = combinations[row, 1]
    q = combinations[row, 2]
    knots = knots_pos[[q]]
    X <- remap(d, q, knots, train_x)
    cv.out = cv.glmnet(X, train_y, alpha = alpha, nfolds = 10) 
    bestlam = cv.out$lambda.min
    mod = glmnet(X, train_y, alpha = 0, lambda = bestlam)
    y_pred = predict(mod, newx = X)
    
    rmse[row] <- sqrt(mean((train_y - y_pred)^2))
    features <- append(features, list(X))
  }
  idx_min <- combinations[which.min(rmse),]
  return(list(idx_min, features, rmse, knots_pos))
}

## Quantile based knots -------

set.seed(123)
res = bestmodel_pos(10, 3, train$x, train$y, 1, 'quantile')

idx_min = res[[1]]
d = idx_min$Var1
q = idx_min$Var2
X = res[[2]][[as.integer(rownames(idx_min))]]
knots = res[[4]][[q]]

Xtest = remap(d, q, knots, test$x)

train_data <- data.frame(cbind(X, "y" = train$y))
test_data <- data.frame(Xtest)
colnames(train_data) <- c(paste0('X', 2:ncol(train_data)-1), 'y')
mod <- lm(y ~ ., data = train_data)
y_pred <- predict(mod, newdata = test_data) 

plot(x, y)
points(test$x, y_pred, col = "red")

## Maximum curvature-based knots ----------

library(Thermimage)

prova <- slopeEveryN(train$x, n = 2)

# install.packages("pracma")
library(pracma)

library(numDeriv)

grad_norms <- sapply(x, function(x) sqrt(sum(gradient(train_data, x = x)^2)))

gradient(train$y)

# Calculate the curvatures
curvature <- curvatures(x, y)

# Find the x values of the local maximum curvatures
max_curv <- x[which.max(curvature)]

## Hierarchical clustering knots ----------

# Hierarchical clustering is a useful technique to choose knots
# for spline regression. The basic idea is to cluster the data points
# based on their pairwise distance and choose the cluster centers as
# the knots for the spline.

# Perform hierarchical clustering on the data
d <- dist(train$y)
hc <- hclust(d)

# Determine the number of clusters
k <- 10
hc_labels <- cutree(hc, k = k)
hc_labels
# Choose cluster centers as knots for the spline
knots <- numeric(k)
for (i in 1:k) {
  knots[i] <- mean(train$x[hc_labels == i])
}

knots <- unlist(lapply(1:k, function(i)mean(train$x[hc_labels == i])))

# Fit the spline with the chosen knots
X <- remap(3, k, knots, train$x)
mod <- lm(train$y ~ X)
y_pred <- predict(mod)

# Plot the results
plot(train$x, train$y, main = "Spline with Hierarchical Clustering Knots")
points(train$x, y_pred, col = "red")
abline(v=knots)
sort(knots)


## Function -----------

# pos can assume values equi (default), quantile, maxcurve?, cluster
bestmodel_pos <- function(qmax=1, dmax=1, train_x, train_y, alpha=0, pos = 'equi'){
  D <- 1:dmax
  Q <- 1:qmax
  combinations <- data.frame(expand.grid(D, Q))
  rmse <- c(rep(NA, nrow(combinations)))
  knots_pos <- list()
  features <- list()
  if(pos == 'equi'){
    knots_pos = lapply(Q, function(q)seq(1/(q+1), 1 - 1/(q+1), 1/(q+1)))
  }
  if(pos == 'quantile'){
    knots_pos = lapply(Q, function(q) unname(quantile(train_x,
                                                      probs = seq(1/(q+1), 1 - 1/(q+1), 1/(q+1)))))
  }
  for(row in 1:nrow(combinations)){
    d = combinations[row, 1]
    q = combinations[row, 2]
    if(pos == 'cluster'){
      dist <- dist(train_y)
      hc <- hclust(dist)
      hc_labels <- cutree(hc, k = q)
      knots <- sort(unlist(lapply(1:q, function(i) mean(train_x[hc_labels == i]))))
      knots_pos[[q]] <- knots
    }
    else{
      knots <- knots_pos[[q]]
    }
    X <- remap(d, q, knots, train_x)
    alpha_grid <- seq(0, 1, by=0.1)
    cv.out = cv.glmnet(X, train_y, alpha = alpha, nfolds = 5) 
    bestlam = cv.out$lambda.min
    mod = glmnet(X, train_y, alpha = alpha, lambda = bestlam)
    y_pred = predict(mod, newx = X)
    rmse[row] <- sqrt(mean((train_y - y_pred)^2))
    features <- append(features, list(X))
  }
  idx_min <- combinations[which.min(rmse),]
  return(list(idx_min, features, rmse, knots_pos))
}

# instead of getting the values which minimize (close to min define a closeness)
# TOLERANCE
res = bestmodel_pos(30, 5, train$x, train$y, 0.5, 'cluster')

idx_min = res[[1]]
d = idx_min$Var1
q = idx_min$Var2
X = res[[2]][[as.integer(rownames(idx_min))]]
knots = res[[4]][[q]]
rmse = res[[3]][142]

Xtest = remap(d, q, knots, test$x)
train_data <- data.frame(cbind(X, "y" = train$y))
test_data <- data.frame(Xtest)
colnames(train_data) <- c(paste0('X', 2:ncol(train_data)-1), 'y')

mod <- lm(y ~ ., data = train_data)
y_pred <- predict(mod, newdata = test_data)
y_train <- predict(mod)

plot(train$x, train$y)
points(train$x, y_train, col = "blue")
points(test$x, y_pred, col = "red")
abline(v=knots)

file <- read.csv('submission1.csv')
file$target_new <- y_pred
file$x <- test$x

abline(h = seq(-2000, 10000, 1000))
abline(v = seq(0, 1, 0.1), col = "red")

# Nested CV -----------------


# Ссылка на датасет: https://archive.ics.uci.edu/ml/datasets/adult
library(caret)
library(xgboost)
library(gbm)
library(randomForest)


data <- read.table("adult.txt", header=T, stringsAsFactors=FALSE)
data_control <- read.table("AdultTest2.txt", stringsAsFactors=FALSE)

colnames(data_control) <- colnames(data)

# Шаг 0. Проверка на пропущенные значения и одинаковость набор значений в признаках:
summary(data)
summary(data_control)
# Как видно из описания оба датасета не имеют пропущенные значения и имеют в каждом из признаков
# по одинаковому набору значений


# Шаг 1. Разбивка выборки data на X_train, y_train, X_test, y_test с учетом стратификации и выделение y_control, X_control:

set.seed(1234)
trainIndex <- createDataPartition(data$pred, p = .7, times = 1, list = F)
y_train <- data$pred[trainIndex]
y_test <- data$pred[-trainIndex]
y_control <- data_control$pred
X_train <- data[,-15][trainIndex,]
X_test <- data[,-15][-trainIndex,]
X_control <- data_control[,-15]

# Используем различные алгоритмы классификации:

# 1. Логистическая классификация:
# 1.1 Маштабируем количественные признаки:
# Разделяем на количественные и вещественные признаки нашу тестовую, обучающую выборку и контрольную выборки:

numeric_cols <- c(1, 3, 5, 11, 12, 13)
X_train_num <- X_train[, numeric_cols]
X_train_cat <- X_train[, -numeric_cols]
X_test_num <- X_test[, numeric_cols]
X_test_cat <- X_test[, -numeric_cols]
X_control_num <- X_control[, numeric_cols]
X_control_cat <- X_control[, -numeric_cols]

# Маштабируем количественные признаки:

X_train_num <- scale(X_train_num, scale = T, center = T)
X_test_num <- scale(X_test_num, scale=T, center=T)

# Кодируем качественные признаки с помощью подхода one-hot encoding:


dmy <- dummyVars("~.", data = X_train_cat)
X_train_cat_encoded <- data.frame(predict(dmy, newdata = X_train_cat))
X_test_cat_encoded <- data.frame(predict(dmy, newdata = X_test_cat))
X_control_cat_encoded <- data.frame(predict(dmy, newdata = X_control_cat))

diff <- setdiff(colnames(X_train_cat_encoded), colnames(X_test_cat_encoded))
diff2 <- setdiff(colnames(X_train_cat_encoded), colnames(X_control_cat_encoded))

for (k in 1:length(diff)){
  X_test_cat_encoded[,diff[k]] <- NA
}

# Соединяем обратно обучающую и тестовую выборку:

X_train2 <- cbind(X_train_num, X_train_cat_encoded)
X_test2 <- cbind(X_test_num, X_test_cat_encoded)
X_control2 <- cbind(X_control_num, X_control_cat_encoded)

# Заменяем в переменной y_train  и y_test "<=50K" на 0 и ">50K" на 1:
len <- length(y_train[which(y_train == "<=50K")])
y_train[which(y_train == "<=50K")] <- 0
y_train[which(y_train == ">50K")] <- 1

len2 <- length(y_test[which(y_test == "<=50K")])
y_test[which(y_test == "<=50K")] <- 0
y_test[which(y_test == ">50K")] <- 1

len3 <- length(y_control[which(y_test == "<=50K.")])
y_control[which(y_control == "<=50K.")] <- 0
y_control[which(y_control == ">50K.")] <- 1

# 1.2 Инициализиурем класс xgboost с базовым алгоритмом в виде логистической регрессии:

estimator_logistic <- xgboost(data = data.matrix(X_train2[ ,sort(names(X_train2))]), label = y_train, class.stratify.cv = T,
                              eta = 0.1, max.depth = 5, objective = "binary:logistic", nrounds = 100)
pred <- predict(estimator_logistic ,data.matrix(X_test2[ ,sort(names(X_train2))]), na.action = na.omit)
pred[pred >= 0.5] <- 1
pred[pred < 0.5] <- 0
error_logistic1 <- sum(pred != y_test)/length(y_test)
cat("Ошибка xgb_logistic на тесте =", error_logistic1)

# Попытаемся улучшить результат с помощью подбора двух гиперпараметров(кол-во деревьев и их глубина):
errorlogictic <- function(predict_proba, test_labels){
  predict_proba[predict_proba >= 0.5] <- 1
  predict_proba[predict_proba < 0.5] <- 0
  return (sum(predict_proba != test_labels)/length(test_labels))
}

xgb_tree_train <- list()
xgb_tree_test <- list()
for (ntree in seq(100, 1000, 100)){
  estimator1 <- xgboost(data = data.matrix(X_train2[ ,sort(names(X_train2))]), label = y_train, class.stratify.cv = T,
                        eta = 0.1, max.depth = 5, objective = "binary:logistic", nrounds = ntree, verbose = 0)
  xgb_tree_train <- c(xgb_tree_train, tail(estimator1$evaluation_log, n = 1))
  predictions <- predict(estimator1 ,data.matrix(X_test2[ ,sort(names(X_train2))]), na.action = na.omit)
  xgb_tree_test <- c(xgb_tree_test, errorlogictic(predictions, y_test))
}


plot(seq(100, 1000, 100), unlist(xgb_tree_train, use.names = F)[seq(2, length(unlist(xgb_tree_train, use.names = F)), 2)], type = "l", col = "blue", xlab = "Количество деревьев", ylab = "Ошибка")
par(new = T)
plot(seq(100, 1000, 100), unlist(xgb_tree_test), type = "l", col = "orange", xlab = "Количество деревьев", ylab = "Ошибка", axes = F, main = "Зависимость ошибки классификации xgboost_logist от кол-ва деревьев")



# Как видно из графиков оптимум достигается при 200 деревьях.

# Определим оптимальное глубину деревьев:

xgb_tree_train2 <- list()
xgb_tree_test2 <- list()
for (n in seq(2, 15, 2)){
  estimator2 <- xgboost(data = data.matrix(X_train2[ ,sort(names(X_train2))]), label = y_train, class.stratify.cv = T,
                        eta = 0.1, max.depth = n, objective = "binary:logistic", nrounds = 200, verbose = 0)
  xgb_tree_train2 <- c(xgb_tree_train2, tail(estimator2$evaluation_log, n = 1))
  predictions2 <- predict(estimator2 ,data.matrix(X_test2[ ,sort(names(X_train2))]), na.action = na.omit)
  xgb_tree_test2 <- c(xgb_tree_test2, errorlogictic(predictions2, y_test))
}

plot(seq(2, 15, 2), unlist(xgb_tree_train2, use.names = F)[seq(2, length(unlist(xgb_tree_train2, use.names = F)), 2)], type = "l", col = "blue", xlab = "Глубина", ylab = "Ошибка")
par(new = T)
plot(seq(2, 15, 2), unlist(xgb_tree_test2), type = "l", col = "orange", xlab = "Глубина", ylab = "Ошибка", axes = F, main = "Зависимость ошибки классификации xgboost_logist от глубины деревьев")

# Как видно из графика оптимум достигается при глубине деревьев = 4.
# Определим качество на контрольном датасете:

estimator_logistic_final <- xgboost(data = data.matrix(X_train2[ ,sort(names(X_train2))]), label = y_train, class.stratify.cv = T,
                                    eta = 0.1, max.depth = 4, objective = "binary:logistic", nrounds = 200)

pred <- predict(estimator_logistic_final ,data.matrix(X_control2[ ,sort(names(X_train2))]), na.action = na.omit)
pred[pred >= 0.5] <- 1
pred[pred < 0.5] <- 0
error_logistic2 <- sum(pred != y_control)/length(y_control)
cat("Ошибка xgb_logistic на контрольных данных =", error_logistic2)

# Случайный лес:
# Закодируем категориальные признаки:

dmy2 <- dummyVars("~.", data = data[,-numeric_cols])
X_train_cat_encoded <- data.frame(predict(dmy, newdata = data[,-numeric_cols]))
X_test_cat_encoded <- data.frame(predict(dmy, newdata = data_control[,-numeric_cols]))

X_train3 <- cbind(X_train_cat_encoded, data[, numeric_cols])
X_test3 <- cbind(X_test_cat_encoded, data_control[, numeric_cols])

y_train3 <- as.factor(data[,15])
y_test3 <- as.factor(data_control[,15])

rf.res <- randomForest(X_train3, y_train3, ntree=400, mtry=floor(sqrt(ncol(X_train)-1)),
                       replace=TRUE, nodesize = 5, importance=TRUE, localImp=FALSE,
                       proximity=FALSE, norm.votes=TRUE, do.trace=400/10,
                       keep.forest = T, sampsize = floor(0.83*nrow(X_train3)),
                       corr.bias=FALSE, keep.inbag=FALSE, na.action = na.omit)





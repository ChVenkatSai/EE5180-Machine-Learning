#!/usr/bin/env python
# coding: utf-8

# # Problem 1: Bayes Classifier
# 
# Bayes classifiers fall under the class of **generative classifiers**. Generative classifiers attempt to learn the generation process of a dataset, usually by making some assumptions about the process that generates the data. Then such classifiers use the learned model to make a prediction or classify the unseen data. A simple example is a Naïve Bayes Classifier.

# ### Naïve Bayes classifier
# Consider a dataset $\left\{X^{(i)}, Y^{(i)}\right\}_{i=1}^{m}$. Each $X^{(i)}$ is an $n-$dimensional vector of input features. Let $Y^{(i)} \in \{0,1\}$ denote the class to which $X^{(i)}$ belongs (this can be easily extended to multi-class problems as well). A good classifier has to accurately predict the probability that any given input $X$ falls in class $1$ which is $ P(Y=1 | X)$. 
# 
# Recall Bayes theorem,
# 
# \begin{align}
# P(Y|X) &= \frac{P(X|Y)P(Y)}{P(X)} \\
#        &= \frac{P(X_1, X_2, \dots, X_n | Y)P(Y)}{P(X_1, X_2, \dots, X_n)}\\
# \end{align}
# 
# **We use the assumption that features are independent of each other. That is one particular feature does not affect any other feature. Of course these assumptions of independence are rarely true, which is why the model is referred as the "Naïve Bayes" model. However, in practice, Naïve Bayes models have performed surprisingly well even on complex tasks, where it is clear that the strong independence assumptions are false.**
# 
# The independence assumption reduces the conditional probability expression to
# \begin{align}
# P(Y|X) &= \frac{P(X_1 | Y)P(X_2 | Y) \dots P(X_n | Y)P(Y)}{P(X_1)P(X_2)\dots P(X_n)}\\
# \end{align}
# 
# The terms $P(X_i|Y)$ and $P(X_i)$ can be easily estimated/learned from the dataset. Hence, the value of $P(Y|X)$ can be found for each value of $Y$. Finally, the class to which $X$ belongs is estimated as $arg\max_{Y}P(Y|X)$. Moreover since $X$ is independent of $Y$, it is only required to find $arg\max_{Y}P(X|Y)P(Y).$ For better understanding with an example refer [this](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c) article.
# 

# ### Problem statement and Dataset
# In this problem, you would implement, train and test a Naïve Bayes model to learn to classify sentiment (positive/negative) of a given text. The training data is in `all_sentiment_shuffled.txt` file.  You can use the function given below to read the dataset
# 

# In[15]:


def read_corpus(corpus_file):
    """ This function reads the file in the location specified by string 
    `corpus_file` and returns a list of tuples (list of words in text, label)
    """
    out = []
    with open(corpus_file, encoding="utf8") as f:
        for line in f:
            tokens = line.strip().split()
            out.append((tokens[3:], tokens[1]))
    return out


# In[16]:


corpus = read_corpus('all_sentiment_shuffled.txt')
print("Example:\n", " Text: ", corpus[0], "\n  Label: ", corpus[0][1])
print("Total number of documents =", len(corpus[0][0]))


# ### Preprocessing a text document
# We can guess that not all the words in a document will be helpful in classification. The words such as "a", "the", "is", etc appear in all the documents randomly and can be neglected or removed. Also a same word can be written in different tenses while conveying the same mood (example "rot"/"rotten"). Hence the documents need to be preprocessed before using them for training the classifier.
# 
#  Libraries such as `gensim`, `nltk` contain functions for doing these preprocessing steps, and you are welcome to use such functions in your code. Formally, these are the preprocessings to be done to the input text to make them simpler and which can improve the performance of your model as well.
# * **Tokenization**: 
#     1.   Split the text into sentences and the sentences into words
#     2.   Lowercase the words and remove punctuation
# * Remove all **stopwords** (stopwords are commonly used word such as "the", "a", "an", "in")
# * Remove all words that have fewer than 3 characters.
# * **Lemmatize** the document (words in third person are changed to first person, and verbs in past and future tenses are changed into present).
# 

# In[18]:


#pip install gensim

""" Implement preprocessing functions here. Use the python modules named above 
for implementing the functions. 
"""
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Removes all the punctuations present in the document
def remove_punctuation(doc):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in doc:
        if char not in punctuations:
            no_punct = no_punct + char

    return no_punct
    pass

# Removes words like 'if', 'he', 'she', 'the', etc which never belongs to any topic
def remove_stopwords(doc):
    # implement
    from gensim.parsing.preprocessing import remove_stopwords
    doc = remove_stopwords(doc)
    return doc
    # comment the next line out
    pass

# lemmatizer is a transformers which transforms the word to its singular, present-tense form
def lemmatize(doc):
    # implement
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    lemmatizer = WordNetLemmatizer()
    docu = word_tokenize(doc)
    result = []
    for word in docu:
        result.append(lemmatizer.lemmatize(word))
    doc = ' '.join(result)
    return doc
    # comment the next line out
    pass

def preprocess(doc):
    """ Function to preprocess a single document
    """
    assert isinstance(doc, str) # assert that input is a document and not the corpus
    processed_doc = remove_punctuation(doc)
    processed_doc = remove_stopwords(processed_doc)
    processed_doc = lemmatize(processed_doc)
    return processed_doc

doc = ' '.join(corpus[0][0])
print(preprocess(doc))

data = []
corp = []
for i in range(len(corpus)):
    data.append(preprocess(str(corpus[i][0])).strip().split())
    corp.append((data[i], corpus[i][1]))


# ### Implementation of Naïve Bayes 
# 
# You can refer the Naïve Bayes section in [this](https://web.stanford.edu/~jurafsky/slp3/slides/7_NB.pdf) slides (slide #32 has a simple pseudo code) to get a hint about implementation of Naïve Bayes for text classification. Then complete the following functions `train_nb` and `classify_nb`.
# 
# NOTE: If you multiply many small probabilities you may run into problems with numeric precision: the probability becomes zero. To handle this problem, it is recommended that you compute the logarithms of the probabilities instead of the probabilities.

# In[19]:


from collections import defaultdict
import pylab as py

def train_nb(training_documents):
    # returning data we need to classify new instances
    #counting pos and neg statement & counting each word in pos and neg statements 
    t_pos = 0
    t_neg = 0
    pos_w = {}
    neg_w = {}        

    for i in range(len(training_documents)):
        if (training_documents[i][1] == 'pos'):    #total words in pos statements
            for j in range(len(training_documents[i][0])):
                if (training_documents[i][0][j] in pos_w):
                    pos_w[training_documents[i][0][j]] += 1
                else:
                    pos_w[training_documents[i][0][j]] = 1
                t_pos += 1
                
        if (training_documents[i][1] == 'neg'):    #total words in neg statements
            for j in range(len(training_documents[i][0])):
                if (training_documents[i][0][j] in neg_w):
                    neg_w[training_documents[i][0][j]] += 1
                else:
                    neg_w[training_documents[i][0][j]] = 1
                t_neg += 1
                
    return [pos_w,neg_w,t_pos,t_neg]
    # comment the next line out
    pass

def classify_nb(training_documents, training_output):
    # return the guess of the classifier
    obtained_result = []
    
    for i in range(len(training_documents)):
        #probability of pos and neg statements 
        t_pos = training_output[2]/(training_output[2]+training_output[3]) 
        t_neg = py.log(1-t_pos)
        t_pos = py.log(t_pos)
        for word in training_documents[i][0]:
            #multiplying(log) with the cond probability given pos
            if word in training_output[0]:
                t_pos += py.log((training_output[0][word]+1)/(training_output[2]+len(training_output[0]))) 
            else:
                t_pos += py.log(1/len(training_output[0]))
            #multiplying(log) with the cond probability given neg
            if word in training_output[1]:
                t_neg += py.log((training_output[1][word]+1)/(training_output[3]+len(training_output[1])))
            else:
                t_neg += py.log((1/len(training_output[1])))
        # higher prob will be the result
        if(t_pos>t_neg):
            obtained_result.append('pos')
        else:
            obtained_result.append('neg')
            
    return obtained_result
    # comment the next line out
    pass


# ### Train-test split
# After reading the dataset, you must split the dataset into training ($80\%$) and test data ($20\%$). Use training data to train the Naïve Bayes classifier and use test data to check the accuracy.

# In[20]:


import numpy as np
N = int(0.8*len(corp))

training_output = train_nb(corp[:N])

obtained_result = np.array(classify_nb(corp[N:],training_output))
real_result = np.array([ result[1] for result in corp[N:]])

#assert len(result) == len(data_results)
accuracy = np.sum(real_result==obtained_result)
print('Accuracy :' ,(accuracy)/len(obtained_result)*100,'%')


# ### Comparison (Bonus)
# Also use `sklearn`'s Naïve Bayes classifier and compare its performance with the classifier you implemented. 

# In[ ]:





# Make sure your code is well documented with comments explaining everything done in your algorithm. With this being said, you are free to design your code anyway you like as long as it implements a Naïve Bayes model and is easily understandable. If you digress from the given code template, explain briefly the structure of your code as well.

# # Problem 2: Regularization and bias-variance trade-off
# 

# ### Problem statement
# In this question we will see how regularization can be used to prevent overfitting of data and then observe the bias-variance tradeoff in a practical setting.

# ### Dataset generation
# - Generate 10 data points $f(x)=sin(2\pi x)$ where $x \hspace{0.1cm} \epsilon \hspace{0.1cm} [0, 1]$ is sampled uniformly.
# - Add Gaussian noise $N(0, 0.5)$ to the generated data. By generating data in this way, we are capturing a property of many real data sets - namely, that they possess an underlying regularity $f(x)$, which we wish to learn, but that individual observations are corrupted by random noise $N(0,0.5)$.
# - We will now use this set of 10 data points as the training dataset.

# In[21]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


Xt = np.linspace(0, 1, num=10).reshape(-1, 1)
yt = (np.sin(2*np.pi*Xt) + np.random.normal(0,0.5,(10,1))).reshape(-1,1)


# ### Polynomial curve fitting 
# - Fit 5 polynomial regression models with varying polynomial orders $M = \{0, 1, 3, 6, 9\}$ on the training dataset. Use the polynomial function of the form:  $y(x, \textbf{w})=\sum^{M}_{j=0}w_jx^j$ and $L2$ loss as the error function: $E(\textbf{w})= \frac{1}{2}\sum^{N}_{n=1}\{y(x_n, \textbf{w}) - t_n)\}^2$, where $t_n$ is the true output for the input $x_n$, and $N$ is the total number of training points.
# - For each model: $M = \{0, 1, 3, 6, 9\}$, plot the graph of the function obtained from fitting the model onto the training dataset along with the training dataset points. 
# - Report the mean squared error on the training dataset and explain its trend with increasing model complexity. Comment on overfitting and underfitting.
# - For each model: $M = \{0, 1, 3, 6, 9\}$, report the coefficients $\textbf{w}^*$. Explain the trend in the coefficients with increasing model complexity. 
# - The goal here is to achieve good generalization by making
# accurate predictions for new data, and not the training data. Use the data generation procedure used previously to generate 100 data points but with new choices for the random noise values included in the target values. These 100 data points will now form our validation dataset.
# - Evaluate each model: $M = \{0, 1, 3, 6, 9\}$ on the validation set and report the mean squared error for each model. 
# - Plot the training and validation set mean squared errors for models with $M = \{0, 1, 3, 6, 9\}$ on the same graph. Explain the trend in the error values with increasing model complexity.

# In[70]:


def make_dataset(F,k):
    F_0 = np.ones(k*1).reshape(k,1)
    F_1 = np.ones(2*k).reshape(k,2)
    F_3 = np.ones(4*k).reshape(k,4)
    F_6 = np.ones(7*k).reshape(k,7)
    F_9 = np.ones(10*k).reshape(k,10)
  
    for i in range(k) :
        F_1[i][1] = F_1[i][1] + F[i] -1
    for i in range(k) :
        F_3[i][1] = F_3[i][1] + F[i] -1
        F_3[i][2] = F_3[i][2] + pow(F[i],2) -1 
        F_3[i][3] = F_3[i][3] + pow(F[i],3) -1
    for i in range(k) :
        F_6[i][1] = F_6[i][1] + pow(F[i],1) -1
        F_6[i][2] = F_6[i][2] + pow(F[i],2) -1
        F_6[i][3] = F_6[i][3]+ pow(F[i],3) -1
        F_6[i][4] = F_6[i][4] + pow(F[i],4) -1
        F_6[i][5] = F_6[i][5] + pow(F[i],5) -1
        F_6[i][6] = F_6[i][6] + pow(F[i],6) -1
    for i in range(k) :
        F_9[i][1] = F_9[i][1] + pow(F[i],1) -1
        F_9[i][2] = F_9[i][2] + pow(F[i],2) -1
        F_9[i][3] = F_9[i][3] + pow(F[i],3) -1
        F_9[i][4] = F_9[i][4] + pow(F[i],4) -1
        F_9[i][5] = F_9[i][5] + pow(F[i],5) -1
        F_9[i][6] = F_9[i][6] + pow(F[i],6) -1
        F_9[i][7] = F_9[i][4] + pow(F[i],7) -1
        F_9[i][8] = F_9[i][5] + pow(F[i],8) -1
        F_9[i][9] = F_9[i][6] + pow(F[i],9) -1

    return F_0,F_1,F_3,F_6,F_9

def regressor(F_0,F_1,F_3,F_6,F_9,y):
    regress0 = LinearRegression().fit(F_0, y)
    regress1 = LinearRegression().fit(F_1, y)
    regress3 = LinearRegression().fit(F_3, y)
    regress6 = LinearRegression().fit(F_6, y)
    regress9 = LinearRegression().fit(F_9, y)

    return regress0, regress1, regress3, regress6, regress9

def predictor(k,F_0,F_1,F_3,F_6,F_9,y,regress0, regress1, regress3, regress6, regress9):  
    y_pred = np.empty(5*k).reshape(5,k,1)

    y_pred[0] = regress0.predict(F_0)
    y_pred[1] = regress1.predict(F_1)
    y_pred[2] = regress3.predict(F_3)
    y_pred[3] = regress6.predict(F_6)
    y_pred[4] = regress9.predict(F_9)

    return y_pred 

def plotter(y_pred,F,Y):
    plt.plot(F,y_pred[0], color="red",label='Degree = 0')
    plt.plot(F,y_pred[1], color="green",label='Degree = 1')
    plt.plot(F,y_pred[2], color="yellow",label='Degree = 3')
    plt.plot(F,y_pred[3], color="blue",label='Degree = 6')
    plt.plot(F,y_pred[4], color="brown",label='Degree = 9')
    plt.title("Polynomialfit for various degrees")
    plt.Flabel("F")
    plt.ylabel("f(F)")
    plt.legend()
    plt.scatter(F,Y)
    plt.show()

F_0t,F_1t,F_3t,F_6t,F_9t = make_dataset(Ft,10)
regress0t, regress1t, regress3t, regress6t, regress9t = regressor(F_0t,F_1t,F_3t,F_6t,F_9t,yt)
y_predt  = predictor(10,F_0t,F_1t,F_3t,F_6t,F_9t,yt,regress0t, regress1t, regress3t, regress6t, regress9t)
plotter(y_predt,Ft,yt)

mset = np.zeros(5).reshape(5,1)
for i in range(5): mset[i] = (((yt - y_predt[i]) ** 2).mean(aFis=0))/2 
print("\n\nMean Squared Errors for different degrees for training set:","\nDegree0", mset[0][0]
      ,"\nDegree1", mset[1][0],"\nDegree3", mset[2][0],"\nDegree6", mset[3][0],"\nDegree9", mset[4][0])


w_0t = regress0t.coef_
w_1t = regress1t.coef_
w_3t = regress3t.coef_
w_6t = regress6t.coef_
w_9t = regress9t.coef_

print("\n\nw* for degree 0 is", w_0t[0])
print("w* for degree 1 is", w_1t[0])
print("w* for degree 3 is", w_3t[0])
print("\nw* for degree 6 is", w_6t[0])
print("\nw* for degree 9 is\n", w_9t[0])

Fv = np.linspace(0, 1, num=100).reshape(-1, 1)
yv = (np.sin(2*np.pi*Fv) + np.random.normal(0,0.5,(100,1))).reshape(-1,1)

F_0v,F_1v,F_3v,F_6v,F_9v = make_dataset(Fv,100)
y_predv = predictor(100,F_0v,F_1v,F_3v,F_6v,F_9v,yv,regress0t, regress1t, regress3t, regress6t, regress9t)
plotter(y_predv,Fv,yv)

msev = np.zeros(5).reshape(5,1)
for i in range(5): msev[i] = (((yv - y_predv[i]) ** 2).mean(aFis=0))/2
print("\n\nMean Squared Errors","\nDegree0", msev[0][0]
      ,"\nDegree1", msev[1][0],"\nDegree3", msev[2][0],"\nDegree6", msev[3][0],"\nDegree9", msev[4][0])


error1 = []
for i in range(5): error1.append(mset[i][0])
error2 = []
for i in range(5): error2.append(msev[i][0])
M = [0,1,3,6,9]

plt.plot(M,error1,label = "Training", marker = "s")
plt.plot(M,error2,label = "Validation", marker ="o")
plt.Flabel("Degree of Model")
plt.ylabel("Mean square error")
plt.Fticks(M,M)

plt.show()


# ### Regularization
# 
# We've seen the effects of increasing model complexity on the training error and the validation error above. We will now use L2 regularization to reduce overfitting.
# 
# - Fit a polynomial regression model of order $M=9$ to the same training dataset as before but now using the regularized error function given by $E^{'}(\textbf{w})= \frac{1}{2}\sum^{N}_{n=1}\{y(x_n, \textbf{w}) - t_n)\}^2 + \frac{\lambda}{2}{\|\textbf{w}\|}^2$ where $\lambda$ is the regularization hyperparameter. Use the following values for $\lambda$: $\lambda={0.01, 0.1, 1}$.
# - Report the coefficients of the model fit above for $\lambda={0.01, 0.1, 1}$. Explain the trend in the coefficient values with increasing $\lambda$.
# - Find the optimal value of the hyperparameter $\lambda$. 
# - Compare the validation error results of the following two models : polynomial regression model of order $M=9$ without regularization and polynomial regression model of order $M=9$ with regularization hyperparameter as estimated above.

# In[66]:


from sklearn.linear_model import Ridge

def ridge_lambda(k,F,y):
    ridge_9 = Ridge(alpha = k)
    ridge_9.fit(F,y)
    coef = ridge_9.coef_
    return ridge_9

ridge_001 = ridge_lambda(0.01, F_9t, yt)
ridge_01 = ridge_lambda(0.1, F_9t, yt)
ridge_1 = ridge_lambda(1, F_9t, yt)

print("Coefficients:\n")
print(ridge_001.coef_[0])
print(ridge_01.coef_[0])
print(ridge_1.coef_[0])



def error(k,F,y):
    ridge_9 = Ridge(alpha = k)
    ridge_9.fit(F,y)
    y_predict = ridge_9.predict(F)
    coef = ridge_9.coef_
    error_lin = ((y-y_predict)**2).mean(aFis=0)
    error_lambda = 0
    for j in range(10):
        error_lambda = error_lambda + abs(coef[0][j]) 
        error_final = (error_lin + k*error_lambda)/2
    return error_final

def error_range(k,m):
    for i in range(0,k,m):
        ridge_9 = Ridge(alpha = i)
        ridge_9.fit(F_9t,yt)
        y_predict = ridge_9.predict(F_9t)
        coef = ridge_9.coef_
        error_lin = ((yt-y_predict)**2).mean(aFis=0)
        error_lambda = 0
    for j in range(10):
        error_lambda = error_lambda + abs(coef[0][j]) 
    error_final = (error_lin + k*error_lambda)/2
    print("error for lambda = ", i, error_final)

print("error for lambda = ", 0.01,"is", error(0.01,F_9t,yt)[0])
print("error for lambda =  ", 0.1,"is", error(0.1,F_9t,yt)[0])
print("error for lambda =    ", 1,"is", error(1,F_9t,yt)[0])


lambda_ = 0.01
error_polyreg = msev[4][0]

ridge_9 = Ridge(alpha = lambda_)
ridge_9.fit(F_9t,yt)
y_predict = ridge_9.predict(F_9v)
coef = ridge_9.coef_
error_ridge = error(1,F_9v,yv)
error_lin = ((yv-y_predict)**2).mean(aFis=0)
error_lambda = 0
for j in range(10):
    error_lambda = error_lambda + abs(coef[0][j]) 
  #print(error_lambda)
error_final = (error_lin + lambda_*error_lambda)/2
print("error for 9 degree polynomial regression is",error_polyreg)
print("error for 9 degree ridge regression is",error_final[0])
print("error for 9 degree ridge regression is",error_lin[0]/2)


# ### Bias-variance trade-off:
# 
# In class you have seen that the expected prediction error for any model can be decomposed as the sum of $bias^2, variance$ and $irreducible\,noise$. We will now observe the bias-variance trade-off for a polynomial regression model of order $M=9$ with varying regularization hyperparameter.
# - Generate $50$ datasets, each containing $10$ points, independently, from the curve $f(x)=sin(2\pi x)$. Add gaussian noise $N(0,0.5)$ to each data point.
# - Fit a polynomial regression model of order $M=9$ to each training dataset by minimizing the regularized error function $E^{'}(\textbf{w})$ with $\lambda=1$.
# - Plot the following:
#   - function obtained by training the model on each of the 50 datasets in the same figure.
#   - The corresponding average of the 50 fits and the sinusoidal function from which the datasets were generated in the same figure.
# - Repeat this exercise for two more $\lambda$ values: $\lambda$ = 0.1, 10.
# - Based on the plots obtained, explain the trend in the bias and variance values with increasing model complexity.
# - Bonus (optional and will not be graded) : 
#   - Plot the $bias^2$, $variance$  and $bias^2 + variance$ against $\lambda$.
#   - Also plot the average test error on a test data size of 1000 points (generated in a similiar way as the 50 training datasets, but independently) against $\lambda$ on the same figure.
#   - For your reference: 
# $$
# Bias^2= (E_{D}[\hat f(x)] - f(x))^2
# \\
# Variance = E_{D}[(\hat f(x) - E_{D}[\hat f(x)])^2]
# $$
# Here $\hat f$ is the trained model and $D$ is the set of all datasets. Use the $50$ training datasets to compute the empirical estimations.

# In[30]:


Ft = np.linspace(0, 1, num=10).reshape(-1, 1)
yt = (np.sin(2*np.pi*Ft) + np.random.normal(0,0.5,(10,1))).reshape(-1,1)

F = np.zeros(500).reshape(50,10,1)
for i in range(50) : F[i] = Ft 
y = np.sin(2*np.pi*F) + (np.random.normal(0,0.5,500)).reshape(50,10,1)

F_9 = np.ones(5000).reshape(50,10,10)
 
for i in range(10) :
    F_9[0][i][1] = F_9[0][i][1] + pow(F[0][i][0],1) -1
    F_9[0][i][2] = F_9[0][i][2] + pow(F[0][i][0],2) -1
    F_9[0][i][3] = F_9[0][i][3] + pow(F[0][i][0],3) -1
    F_9[0][i][4] = F_9[0][i][4] + pow(F[0][i][0],4) -1
    F_9[0][i][5] = F_9[0][i][5] + pow(F[0][i][0],5) -1
    F_9[0][i][6] = F_9[0][i][6] + pow(F[0][i][0],6) -1
    F_9[0][i][7] = F_9[0][i][4] + pow(F[0][i][0],7) -1
    F_9[0][i][8] = F_9[0][i][5] + pow(F[0][i][0],8) -1
    F_9[0][i][9] = F_9[0][i][6] + pow(F[0][i][0],9) -1

for i in range(50): F_9[i] = F_9[0]

y_predict = np.zeros(500).reshape(50,10,1)


ridge_9 = Ridge(alpha = 1)
for i in range(50): 
    ridge_9.fit(F_9[i],y[i])
    y_predict[i] = ridge_9.predict(F_9[i])



for i in range(50):
    plt.plot(Ft,y_predict[i])
plt.scatter(Ft,yt, label ="f(F)")
plt.show()


y_mean_test = y.reshape(50,10).mean(aFis = 0) 
y_mean_predict = y_predict.reshape(50,10).mean(aFis =0)
plt.plot(Ft,y_mean_predict,label = "mean of predicted values")
plt.plot(Ft,y_mean_test,label = "mean of test values")
plt.legend()


# #Problem 3: Logistic Regression

# ## Binary Logistic Regression
# 
# Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic **sigmoid function** $h_ \theta (\cdot)$ to return a probability value which can then be mapped to two or more discrete classes. $$ h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^{- \theta^Tx} }  $$ 
# 
# <br>here, the vector $\theta$ represents the weights and the vector $x$ represents the given inputs.
# 

# ## Problem 3, Part A: Dataset A
# 

# 
# Use Dataset A (``data_prob3_parta.csv``) for this part of the question. The given CSV file has three columns: column 1 is the first input feature, column 2 is the second input feature and column 3 is the output label. Split the dataset into training data (75%) and testing data (25%) randomly.
# 

# In[31]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

dataset = pd.read_csv('data_prob3_parta.csv')

# input
x = dataset.iloc[:, [0, 1]].values
  
# output
y = dataset.iloc[:, 2].values


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)


# Visualize the training data with a scatter plot (input feature 1 on the X axis, input feature 2 on the Y axis and color the points according to their labels).

# In[32]:




x = xtrain[:, 0]
y = xtrain[:, 1]

plt.scatter(x,y,color=['blue'])
plt.show()


# Build the logistic regression model using the training data. 
# 
# The scikit library can be used to build the model. Bonus marks will be awarded if the model is built from scratch without using any external libraries. If you are writing your own implementation, try to keep number of features and number of classes as variables for next parts.

# In[33]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)


# Print the final weights.

# In[34]:


print(y_pred)


# Print the final accuracy on test data.

# In[35]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))


# Plot the scatter plot on test data. On top of this scatter plot, plot the decision boundaries.

# In[36]:


from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
							stop = X_set[:, 0].max() + 1, step = 0.01),
					np.arange(start = X_set[:, 1].min() - 1,
							stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
			np.array([X1.ravel(), X2.ravel()]).T).reshape(
			X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
	
plt.title('Classifier (Test set)')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()


# ## Problem 3, Part B: Dataset B
# 

# 
# Use Dataset B (``data_prob3_partb.csv``) for this part of the question. The given CSV file has three columns: column 1 is the first input feature, column 2 is the second input feature and column 3 is the output label. Split the dataset into training data (75%) and testing data (25%) randomly.
# 

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

dataset = pd.read_csv('data_prob3_partb (1).csv')

# input
x = dataset.iloc[:, [0, 1]].values
  
# output
y = dataset.iloc[:, 2].values


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)


# Visualize the training data with a scatter plot (input feature 1 on the X axis, input feature 2 on the Y axis and color the points according to their labels).

# In[40]:



x = xtrain[:, 0]
y = xtrain[:, 1]

plt.scatter(x,y,color=['blue'])
plt.show()


# Build the logistic regression model using the training data. The scikit library can be used to build the model. Bonus marks will be awarded if the model is built from scratch without using any external libraries.

# In[41]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)


# Print the final weights.

# In[42]:


print(y_pred)


# Print the final accuracy on test data.

# In[43]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))


# Plot the scatter plot on test data. On top of this scatter plot, plot the decision boundaries.

# In[44]:


from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
							stop = X_set[:, 0].max() + 1, step = 0.01),
					np.arange(start = X_set[:, 1].min() - 1,
							stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
			np.array([X1.ravel(), X2.ravel()]).T).reshape(
			X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
	
plt.title('Classifier (Test set)')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()


# As you can see, a straight line is not the best decision boundary for this type of data. In the next part, we will try polynomial feature mapping to generate more features and build the classifier on top of it.

# ## Problem 3, Part C: Polynomial Feature Mapping
# 

# 
# Use Dataset B (``data_prob3_partb.csv``) for this part of the question.
# 

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os

dataset = pd.read_csv('data_prob3_partb (1).csv')
print(dataset)


# Generate more features for each datapoint using the following transformation.
# 
# For a datapoint $(x_1, x_2)$,
# $$ (x_1, x_2) \rightarrow (x_1, x_2, x_1^2, x_2^2, x_1^3, x_2^3, ..., x_1^T, x_2^T) $$
# Now, instead of giving $(x_1, x_2)$ as the input to the classifier, use the transformed data as the input to the classifier.

# Generate the transformed training and testing dataset using Dataset B (``data_prob3_partb.csv``).

# In[46]:


def add_transform(data,n):
    s=str(n)
    data['x1^'+s]=data['Input feature 1']**n
    data['x2^'+s]=data['Input feature 2']**n
    return data

data=pd.read_csv("data_prob3_partb (1).csv")
X = data[['Input feature 1', 'Input feature 2']]
y = data['Output label']

T=9
for i in range(2,T+1):
    X_1=add_transform(X,i)
print(X)


# Build the logistic regression model using the transformed training data. The scikit library can be used to build the model. Bonus marks will be awarded if the model is built from scratch without using any external libraries.

# In[47]:


# input
x = X_1.iloc[:, :].values
  
# output
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)


# Try different values of $T$ (highest number of degree) between 3 to 10. Find out which value of $T$ gives the best test accuracy. Please print that values of $T$ in the below cell.

# In[48]:


mix = []
for t in range(3,10):
    X = data[['Input feature 1', 'Input feature 2']]
    for i in range(2,t+1):
        X_1=add_transform(X,i)  
    # input
    x = X_1.iloc[:, :].values
    # output
    y = dataset.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(
            x, y, test_size = 0.25, random_state = 0)
    
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(xtrain, ytrain)
    y_pred = classifier.predict(xtest)
    
    from sklearn.metrics import accuracy_score
    print ("Accuracy : ", accuracy_score(ytest, y_pred))
    mix.append(accuracy_score(ytest, y_pred))
    
maxi = mix[0]
T = 3
for k in range(len(mix)):
    if mix[k]>maxi:
        maxi = mix[k]
        T = k+3
print(T)


# Print the final weights.

# In[49]:


X = data[['Input feature 1', 'Input feature 2']]
for i in range(2,T+1):
     X_1=add_transform(X,i) 
# input
x = X_1.iloc[:, :].values
  
# output
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

print(y_pred)


# Print the final accuracy on transformed test data.

# In[50]:


print ("Accuracy : ", accuracy_score(ytest, y_pred))


# Plot the scatter plot on test data (note that this is  the original data , not the transformed one). On top of this scatter plot, plot the new decision boundaries.

# In[53]:


from sklearn import *
from numpy import c_
data = pd.read_csv("data_prob3_partb (1).csv")
# input
x_1 = data.iloc[:, :-1].values
  
# output
y_1 = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
xtrain_1, xtest_1, ytrain_1, ytest_1 = train_test_split(
        x_1, y_1, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain_1, ytrain_1)
y_pred_1 = classifier.predict(xtest_1)

X_set_1, y_set_1 = xtest_1, ytest_1

logistic_regression= LogisticRegression(solver='liblinear', random_state=0)
model=logistic_regression.fit(xtrain,ytrain)

x1=np.linspace(data.values[:,0].min()-1,data.values[:,0].max()+1,500)
x2=np.linspace(data.values[:,1].min()-1,data.values[:,1].max()+1,500)

xx, yy = np.meshgrid(x1,x2)
# flatten each grid to a vector
x1, x2 = xx.flatten(), yy.flatten()
x1, x2 = x1.reshape((len(x1), 1)), x2.reshape((len(x2), 1))
Xv=c_[x1,x2]
df = pd.DataFrame(Xv, columns =['Input feature 1', 'Input feature 2'])
for i in range(2,T+1):
    df=add_transform(df,i)

z=model.predict(df.values)
zz=z.reshape(xx.shape)



plt.contourf(xx, yy, zz, alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set_1[y_set_1 == j, 0], X_set_1[y_set_1 == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
	
plt.title('Classifier (Test set)')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()


# ## Problem 3, Part D: Multi-class Logistic Regression

# ## Multi-class Logistic Regression
# 
# In case of a multi-class classification problem (when the number of classes is greater than two), a **softmax function** is used instead. 
# $$\text{Softmax}(\theta_{i}) = \frac{\exp(\theta_i)}{\sum_{j=1}^{N} \exp(\theta_j)}$$ where $j$ varies from $1$ to $N$ which is the number of classes and  $\theta_{i}$ is $$\theta_{i}=W_{i}*x^{(i)}+b$$ Where $x^{(i)}$ is a feature  vector of dimensions $D \times 1$ and $W_{i}$ is the $i$-th row of the weight matrix $ W$ of  dimensions $N \times D$  and $b$ is the bias having dimensions $D \times 1$.

# 
# Use Dataset D (``data_prob3_partd.csv``) for this part of the question. The given CSV file has three columns: column 1 is the first input feature, column 2 is the second input feature and column 3 is the output label. Split the dataset into training data (75%) and testing data (25%) randomly.
# 

# In[54]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

dataset = pd.read_csv('data_prob3_partd.csv')

# input
x = dataset.iloc[:, [0, 1]].values
  
# output
y = dataset.iloc[:, 2].values


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)


# Visualize the training data with a scatter plot (input feature 1 on the X axis, input feature 2 on the Y axis and color the points according to their labels).

# In[55]:




x = xtrain[:, 0]
y = xtrain[:, 1]

plt.scatter(x,y,color=['blue'])
plt.show()



# Build the logistic regression model using the training data. The scikit library can be used to build the model. Bonus marks will be awarded if the model is built from scratch without using any external libraries.

# In[56]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs',random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)



# Print the final weights.

# In[57]:


print(y_pred)


# Print the final accuracy on test data.

# In[58]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))


# Plot the scatter plot on test data. On top of this scatter plot, plot the decision boundaries.

# In[60]:


from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
							stop = X_set[:, 0].max() + 1, step = 0.01),
					np.arange(start = X_set[:, 1].min() - 1,
							stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
			np.array([X1.ravel(), X2.ravel()]).T).reshape(
			X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
	
plt.title('Classifier (Test set)')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()


# # Problem 4: Learning on real world datasets
# 
# *Cric_data.csv* contains the batting averages and bowling averages of various cricket players along with their role in the team (Bowler/Batsman/Allrounder). The task is to predict the player role based on their batting and bowling averages.

# In the next CodeWrite cell, extract the required columns from the csv file, partition the data into training (75%) and testing (25%) data randomly.  

# In[61]:


# Extract data and partition
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
inp = pd.read_csv('Cric_data.csv', usecols=['Batting Average', 'Bowling Average', 'Player Class'])

# input
x = inp.iloc[:, :-1].values
  
# output
y = inp.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size = 0.25, random_state = 0)

# X_train  
# X_test  
# Y_train  
# Y_test 


# **Binary Classification:**
# 
# Derive the classifiers under the assumptions below, and use ML estimators to compute and return the results on the test set. *Consider only batsmen and bowlers in this part*.
# 
# Let random variable $\underline X$ represent (Batting Average, Bowling Average) of a player whose role is a random variable $Y$.
# 
# 1a) Linear Predictor: Assume $\underline X|Y=Batsman \sim \mathcal{N}(\underline {\mu_-}, I)$ and  $X|Y=Bowler \sim \mathcal{N}(\underline {\mu_+}, I)$. 
# 
# 1b) Bayes Classifier: Assume $\underline X|Y=Batsman \sim \mathcal{N}(\underline {\mu_-}, \Sigma_-)$ and  $X|Y=Bowler \sim \mathcal{N}(\underline {\mu_+}, \Sigma_+)$. 

# In[62]:


import numpy as np
import pylab as py

def mltv_normal(x, d, mean, covariance):#pdf of multivariate dist.
   
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
def Lin_clsf_1a(X_train, Y_train, X_test):#linear classifier binary

    btmn_X = X_train[Y_train== -1].values
    bwl_X = X_train[Y_train== 1].values
    btmn_mean = np.mean(btmn_X,axis = 0)
    bwl_mean = np.mean(bwl_X,axis = 0)
    Py0 = len(Y_train[Y_train == -1].values)/len(Y_train[Y_train!=0].values)
    Py1 = len(Y_train[Y_trai n== 1].values)/len(Y_train[Y_train!=0].values)
    
    predic = []
    for i in range(len(X_test.values)):
        prob0 = mltv_normal(X_test.values[i],2,btmn_mean,np.identity(2))
        prob1 = mltv_normal(X_test.values[i],2,bwl_mean,np.identity(2))
        if(prob0>prob1):
            predic.append(int(-1))
        else:
            predic.append(int(1))
    

    return np.array(predic)


def Bayes_clsf_1b(X_train, Y_train, X_test):#bayes binary
    
    btmn_X = X_train[Y_train ==-1].values
    bwl_X = X_train[Y_train ==1].values
    btmn_mean = np.mean(btmn_X,axis=0)
    bwl_mean = np.mean(bwl_X,axis=0)
    Py0 = len(Y_train[Y_train ==-1].values)/len(Y_train[Y_train!=0].values)
    Py1 = len(Y_train[Y_train ==1].values)/len(Y_train[Y_train!=0].values)
    btmn_cov = np.cov(btmn_X[:,0],btmn_X[:,1])
    bwl_cov = np.cov(bwl_X[:,0],bwl_X[:,1])
    
    predic = []
    for i in range(len(X_test.values)):
        prob0 = mltv_normal(X_test.values[i],2,btmn_mean,btmn_cov)
        prob1 = mltv_normal(X_test.values[i],2,bwl_mean,bwl_cov)
        if(prob0*Py0 > Py1*prob1):
            predic.append(int(-1))
        else:
            predic.append(int(1))


    return np.array(predic)


# **Multi-class Classification:**
# 
# Derive the classifiers under the assumptions below, and use ML estimators to compute and return the results on the test set. *Consider batsmen, bowlers and allrounders in this part*.
# 
# Let random variable $\underline X$ represent (Batting Average, Bowling Average) of a player whose role is a random variable $Y$.
# 
# The $3\times 3$ loss matrix giving the loss incurred for predicting $i$ when truth is $j$ is below. (Ordering: Batsman - Allrounder - Bowler)
# 
# $L=\begin{bmatrix} 0 &1 & 2\\ 1 &0 & 1\\ 2 &1 & 0\end{bmatrix}$ 
# 
# 2a) Linear Predictor: Assume $\underline X|Y=a \sim \mathcal{N}(\underline {\mu_a}, I)$
# 
# 2b) Bayes Classifier: Assume $\underline X|Y=a \sim \mathcal{N}(\underline {\mu_a}, \Sigma_a)$

# In[63]:


def Lin_clsf_2a(X_train, Y_train, X_test):#linear 

    btmn_X = X_train[Y_train == -1].values
    bwl_X = X_train[Y_train == 1].values
    allrounder_X = X_train[Y_train == 0].values
    btmn_mean = np.mean(btmn_X,axis = 0)#calculating various means
    bwl_mean = np.mean(bwl_X,axis = 0)
    allrounder_mean = np.mean(allrounder_X,axis = 0)
    Py_1 = len(Y_train[Y_train == -1].values)/len(Y_train.values)
    Py0 = len(Y_train[Y_train == 0].values)/len(Y_train.values)
    Py1 = len(Y_train[Y_train == 1].values)/len(Y_train.values)
    predic = []
    
    for i in range(len(X_test.values)):
        prob_1 = mltv_normal(X_test.values[i],2,btmn_mean,np.identity(2))
        prob0 = mltv_normal(X_test.values[i],2,allrounder_mean,np.identity(2))#multivar
        prob1 = mltv_normal(X_test.values[i],2,bwl_mean,np.identity(2))
        if(prob0*Py0 >= Py1*prob1 and prob0*Py0 >= Py_1*prob_1):
            predic.append(0)
        if(prob1*Py1 > Py0*prob0 and prob1*Py1 > Py_1*prob_1):
            predic.append(1)
        if(prob_1*Py_1 > Py0*prob0 and prob_1*Py_1 > Py1*prob1):
            predic.append(-1)
    
    return np.array(predic)
            

def Bayes_clsf_2b(X_train, Y_train, X_test):

    btmn_X = X_train[Y_train == -1].values
    bwl_X = X_train[Y_train == 1].values
    allrounder_X = X_train[Y_train == 0].values
    btmn_mean = np.mean(btmn_X,axis = 0)
    bwl_mean = np.mean(bwl_X,axis = 0)
    allrounder_mean = np.mean(allrounder_X,axis = 0)
    Py_1 = len(Y_train[Y_train == -1].values)/len(Y_train.values)
    Py0 = len(Y_train[Y_train == 0].values)/len(Y_train.values)
    Py1 = len(Y_train[Y_train == 1].values)/len(Y_train.values)
    btmn_cov = np.cov(btmn_X[:,0],btmn_X[:,1])
    bwl_cov = np.cov(bwl_X[:,0],bwl_X[:,1])
    allrounder_cov = np.cov(allrounder_X[:,0],allrounder_X[:,1])
    predic = []
    
    for i in range(len(X_test.values)):
        prob_1 = mltv_normal(X_test.values[i],2,btmn_mean,btmn_cov)
        prob0 = mltv_normal(X_test.values[i],2,allrounder_mean,allrounder_cov)
        prob1 = mltv_normal(X_test.values[i],2,bwl_mean,bwl_cov)
        if(prob0*Py0 >= Py1*prob1 and prob0*Py0 >= Py_1*prob_1):
            predic.append(0)
        if(prob1*Py1 > Py0*prob0 and prob1*Py1 > Py_1*prob_1):
            predic.append(1)
        if(prob_1*Py_1 > Py0*prob0 and prob_1*Py_1 > Py1*prob1):
            predic.append(-1)
    
    return np.array(predic)


# **Plots:**
# 
# In the next CodeWrite cell, plot all the 4 classifiers on a 2d plot. Take a suitable grid covering averages (0,60) in both dimensions. (Color the different classes accordingly). Add the training data points also on the plot. Label the plots appropriately. 

# In[73]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('Cric_data.csv', usecols=['Batting Average', 'Bowling Average', 'Player Class'])
X = data[['Batting Average', 'Bowling Average']]
y = data['Player Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from seaborn import scatterplot

xx,yy = np.meshgrid(np.arange(0,60,0.5),np.arange(0,60,0.5))
r1, r2 = xx.flatten(),yy.flatten()
r1, r2 = r1.reshape((len(r1),1)), r2.reshape((len(r2),1))
grid = pd.DataFrame(np.column_stack((r1,r2)),columns = ['batting avg','bowling avg'])
df = pd.concat([X_train[y_train!=0], y_train[y_train!=0]], axis=1, join='inner')
linear_predic1 = Lin_clsf_1a(X_train,y_train,X_test)
linear_db1 = Lin_clsf_1a(X_train,y_train,grid)
zz = linear_db1.reshape(xx.shape)
print('Accuracy with linear predicictor: ',metrics.accuracy_score(y_test.values, linear_predic1))
py.figure(1)
py.contourf(xx,yy,zz)
scatterplot(data = df,x ='Batting Average',y ='Bowling Average',hue ='Player Class',style='Player Class')
py.xlim([0,60])
py.ylim([0,60])
py.show()
bayes_predic1 = Bayes_clsf_1b(X_train,y_train,X_test)
bayes_db1 = Bayes_clsf_1b(X_train,y_train,grid)
zz = bayes_db1.reshape(xx.shape)
print('Accuracy with bayes predicictor: ',metrics.accuracy_score(y_test.values, bayes_predic1))
py.figure(2)
py.contourf(xx,yy,zz)
scatterplot(data = df,x = 'Batting Average',y = 'Bowling Average',hue = 'Player Class',style = 'Player Class')
py.xlim([0,60])
py.ylim([0,60])
py.show()

df=pd.concat([X_train, y_train], axis=1, join='inner')

linear_predic2 = Lin_clsf_2a(X_train,y_train,X_test)
linear_db2 = Lin_clsf_2a(X_train,y_train,grid)
zz = linear_db2.reshape(xx.shape)
print('Accuracy with linear predicictor: ',metrics.accuracy_score(y_test.values, linear_predic2))
py.figure(3)
py.contourf(xx,yy,zz)
scatterplot(data =df,x ='Batting Average',y ='Bowling Average',hue ='Player Class',style ='Player Class')
py.xlim([0,60])
py.ylim([0,60])
py.show()
bayes_predic2 = Bayes_clsf_2b(X_train,y_train,X_test)
bayes_db2 = Bayes_clsf_2b(X_train,y_train,grid)
zz = bayes_db2.reshape(xx.shape)
print('Accuracy with bayes predicictor: ',metrics.accuracy_score(y_test.values, bayes_predic2))
py.figure(4)
py.contourf(xx,yy,zz)
scatterplot(data =df, x='Batting Average',y ='Bowling Average',hue ='Player Class',style ='Player Class')
py.xlim([0,60])
py.ylim([0,60])
py.show() 


# **Observations:**
# 
# In the next Textwrite cell, summarise (use the plots of the data and the assumptions in the problem to explain) your observations regarding the four learnt classifiers, and also give the error rate of the four classifiers as a 2x2 table.

# ** Cell type : TextWrite ** 
# (Write your observations and table of errors here)
# 1. Well in the first one we had a text which was either labelled negative or positive. We had to train it using a naive bayes. First we preprocessed the text by taking out punctuations and stopwords like his her etc and we also converted all the words into the same tense. Then we went ahead and built the bayes model as wrf to slide 32. We simply compared probabilities of each word given its neg or pos and used this to assign probabilities which in turn we used it the test data and predicted neg or pos. 
# 2. We can clearly see that as the model complexity increases the error comes down as expected. As we try to fit a function in a polynomial we get better and better and as we go higher we almost fit it into them and hence error comes down. Similarly with the coefficients of higer degrees. Higer degrees mean a small change in x means a drastic change in the power of that, hence the coefficients are small in order to get a smooth(as possible) graph. As complexity increases the garph goes into overfitting and vice-versa.As lambda increases to reduce the error or to have the same minimum error(check the error expression) the coefficients have to be small hence the coefficients reduce with increasing lamda. Now we go with lambda which is basically used to reduce overfitting. Now as lamda increases, the coefficients reduce leading into underfitting hence the optimal would be the lowest one i.e 0.01. And we also observe that the error is less for the ones with parameter as expected. A very simple model that makes a lot of mistakes is said to have high bias. A very complicated model that does well on its training data is said to have low bias. Negatively correlated with bias is the variance of a model, which describes how much a prediction could potentially vary if one of the predictors changes slightly. In the simple model , the simplicity of the model makes its predictions change slowly with predictor value, so it has low variance. On the other hand, as model complexity increases, low bias model likely fits the training data very well and so predictions vary wildly as predictor values change slightly. This means this model has high variance, and it will not generalize to new data well. Thus is the trade-off between them.
# 3. All the observartions have been mentioned above with the results. We explore and exploit the properties of logistic regression in this. At first we have a simple binary classification problem. And then we see that it can't always be used for binary properly with given features so we extend it by taking the squares and thus by defining a finer boundary for the two classes. And finaaly we also see a multi-class problem with logistic regression.
# 4. We now dive into the real world problems where we assume the probabilities to have the form of gausian. We can see from the above plots that in both the binary and multivariate sections bayes has a greater accuracy than the linear predictor. And the accuracies are less as we assumed a gaussian distribution which gives a spread in the data and hece the lower accuracy. The accuracies(hence the  error) are mentioned above.

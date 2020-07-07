# Advanced Regression
In an earlier module, you had learnt the principles of model selection - model simplicity and complexity, overfitting, regularization etc. In this module, you will learn to apply model selection principles in the regression framework.<br/>
Also, you will learn to extend the linear regression framework to problems which are not strictly 'linear' and understand the difference between linear and nonlinear regression problems.

This module covers the following two concepts:<br/>
1. Generalized Linear Regression
2. Regularized Regression - Ridge and Lasso Regression

## Generalized Regression
In linear regression, you had encountered problems where the target variable y was linearly related to the predictor variables X. But what if the relationship is not linear? Let's see how we can use **generalised regression** to tackle such problems. 

The explanatory and response variables often do not vary in a linear manner, as illustrated in the following examples:
1. In the first example, notice that the data points oscillate and follow a sine or cosine type of function. 

![title](img/Sales_image.png)

2. In the second example of electricity consumption, the data points gradually increase non-linearly, indicative of a polynomial or an exponential function: 

![title](img/Electric_image.png)

You may recall that the general expression of a polynomial function is<br/>
![title](img/Polynomial.JPG)

 If n=2, it is called a quadratic or a second-degree polynomial; if n=3, it is called a cubic or a third-degree polynomial. <br/>
 Also, recall that the **roots of a polynomial** f(x) represent the values of x at which the function cuts the x-axis and that a polynomial function can have both **real or imaginary roots**.
 
 For example, the quadratic function $f(x)=x^2-5x+6$ has two real roots: $x=2,3$ , though the function $f(x)=x^2+2x+10$ does not have any real roots (and does not cut the x-axis). <br/>
 So how can we take the decision of fitting a polynomial model, a sine curve or any other function by just looking at the plot? You'll learn to do that in the next segment.

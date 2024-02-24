"""Machine Learning"""


"""Naive Bayes algorithm:
P(A|B) = P(B|A)*P(A)/P(B)
To get around this problem, we use Naive Bayes, which assumes that all the features 
are independent of each other, so the joint probability is simply the product of 
independent probabilities. This assumption is naive because it is almost always wrong. 
"""

"""Support Vector Machine algorithm:
W_t * X - c = 0
Finding the optimal hyperplane that best segregate the classes.
Each data point in the dataset can be considered a vector in an N-dimensional plane, 
with each dimension representing a feature of the data. SVM identifies the frontier 
data points (or points closest to the opposing class), also known as support vectors, 
and then attempts to find the boundary (also known as the hyperplane in the N-dimensional 
space) that is the farthest from the support vector of each class.
It is obvious that there are many hyperplanes that can segregate the two classes 
in this case. However, the SVM algorithm tries to find the optimum W (coefficients) 
and c (constant) so that the hyperplane is at the maximum distance from both support 
vectors. To perform this optimization, the algorithm starts with a hyperplane with 
random parameters and then calculates the distance of each point from the hyperplane 
using the following equation:
(X_t * X_0 - c) / ||W||"""

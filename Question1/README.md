# Question 1: Factoria

This project has implemented an integrated workflow that combines 
testing and implementation, and the main function of the program 
is to calculate the factorial of an integer.

First, we need to ensure that the data input by the user is definitely 
an integer. If the string entered by the user cannot be interpreted as 
a decimal integer, our program should output the prompt: "Please enter 
an integer!"

Furthermore, if the integer input by the user is not a non-negative 
integer, we need to output the prompt "Please enter a non-negative 
integer!" because factorial is only defined on non-negative integers.

The above function is implemented in the function `MyFactorial`. If the 
input data is invalid, the function returns `None`. Otherwise, the 
function returns the factorial of the input data.

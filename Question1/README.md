# Question 1: Factoria

This project is written in the Python language. Before launching it,
please make sure that you have deployed the Python 3 runtime environment \
on your computer.

## Relavant Files
- Code.py
- TestData.json

## Desciprtion

This project has implemented an integrated workflow that combines 
testing and implementation, and the main function of the program 
is to calculate the factorial of an **integer**.

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

## Test and Run

If you want to manually test the functionality of this program, please 
enter the command `python3 Code.py` in the command line to start the 
program. You just need to enter the corresponding content as prompted 
by the program to perform the test.

If you want to use the automated testing feature of this program, please 
use the command line command `python3 Code.py --test`. The automated 
testing will use the data from the `testdata.json` file.

The code related to testing is implemented in the function `MyAutoTest`. 
Test cases can be expanded by adding data to `testdata.json`.

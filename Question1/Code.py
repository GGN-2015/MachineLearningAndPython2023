import sys

def MyFactorial(x):
    typeError = False
    try:
        x = int(x)
    except:
        print("Please enter an integer!")
        typeError = True
    if not typeError:
        if x < 0:
            print("Please enter a non-negative integer!")
        else:
            ans = 1
            for i in range(1, x + 1):
                ans *= i
            return ans
    return None

if __name__ == "__main__":
    print(MyFactorial(5))

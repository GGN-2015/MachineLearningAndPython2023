import sys
import json

def MyFactorial(x, display_prompt = True):
    typeError = False
    try:
        x = int(x)
    except:
        if(display_prompt):
            print("Please enter an integer!")
        typeError = True
    if not typeError:
        if x < 0:
            if(display_prompt):
                print("Please enter a non-negative integer!")
        else:
            ans = 1
            for i in range(1, x + 1):
                ans *= i
            return ans
    return None

def MyAutoTest(filename):
    # input testdata from file
    assert type(filename) == str
    try:
        with open(filename, "r", encoding="utf8") as f:
            data = json.load(f)
    except:
        print("file not found: <%s>" % filename)
        data = []

    # run test and validation
    assert type(data) == list
    accepted_cases = 0
    wong_cases = 0
    for item in data: # test all test cases
        assert type(item) == dict
        assert item.get("input") is not None
        data_in  = item.get("input")
        data_ans = item.get("answer")
        data_out = MyFactorial(data_in, display_prompt=False)

        if(data_ans == data_out):
            accepted_cases += 1
        else:
            wong_cases += 1
            print("for input: ", data_in, " your output is: ", \
                data_out, "but answer is: ", data_ans)
    print("%d Tested, %d Accepted." % 
        (accepted_cases + wong_cases, accepted_cases))

if __name__ == "__main__":
    if len(sys.argv) == 1: # no command line configuration
        x = input("Please input an integer: ")
        print(MyFactorial(x))

    elif len(sys.argv) == 2 and sys.argv[1] == "--test":
        MyAutoTest('TestData.json')

    else:
        print("Command line configuration unknown.")
        print(sys.argv[1:])

import statistics


import subprocess



for filename in ["corre3.txt", "corre2.txt", "original.txt", "corre2_new_nn.txt"]:

    result = subprocess.run("cat {} | grep Test ".format(filename), shell=True, capture_output=True, text=True)

    # Get the output
    text = result.stdout


    # text = """
    # Test set results, loss = 0.565025, accuracy = 0.702128
    # Test set results, loss = 0.584491, accuracy = 0.723404
    # Test set results, loss = 0.587194, accuracy = 0.702128
    # Test set results, loss = 0.516932, accuracy = 0.734043
    # Test set results, loss = 0.625465, accuracy = 0.648936
    # Test set results, loss = 0.468312, accuracy = 0.819149
    # """

    lines = text.strip().split('\n')
    try: 
        accuracies = [float(line.split()[-1]) for line in lines]

        average = statistics.mean(accuracies)
        variance = statistics.variance(accuracies)
        print('File: {}'.format(filename))
        print('Average: {:.6f}'.format(average))
        print('Variance: {:.6f}'.format(variance))
        print('Max:{:.6f}'.format(max(accuracies)) )

        from time import sleep
        sleep(1)
    except Exception as e:
        print(e)

# Run in parallel 
import multiprocessing
import subprocess
import os
import numpy 

def create_params(run_num, num_iterations):
    run_iter = range(num_iterations)
    run_num = numpy.repeat(run_num, num_iterations)
    param_list = list(zip(run_num, run_iter))
    return param_list

def execute_model_runs(run_num, run_iter):
    print(f'run: {run_num}\titeration: {run_iter}')
    subprocess.call(
        ['python', 
        'C:/Users/cawalden/Documents/GitHub/Pandemic_Model/pandemic/model.py', 
        'C:/Users/cawalden/Documents/GitHub/Pandemic_Model/pandemic/config.json',
        str(run_num), 
        str(run_iter)], 
        shell=True 
    )
    return run_num, run_iter 

if __name__ == '__main__':
    subprocess.call('pipenv install')
    param_list = create_params(100, 1)
    p = multiprocessing.Pool()
    results = p.starmap(execute_model_runs, param_list)
    p.close()

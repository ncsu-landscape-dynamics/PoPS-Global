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
    # run_num = param_list[0]
    # run_iter = param_list[1]
    print(f'run: {run_num}\titeration: {run_iter}')
    subprocess.call(['python', 'C:/Users/cwald/Documents/GitHub/Pandemic_Model/pandemic/model.py', str(run_num), str(run_iter)], shell=True )
    # # print(os.system(f'python model.py {run_num} {run_iter}'))
    return run_num, run_iter 

if __name__ == '__main__':
    param_list = create_params(3, 3)
    p = multiprocessing.Pool()
    results = p.starmap(execute_model_runs, param_list)
    p.close()

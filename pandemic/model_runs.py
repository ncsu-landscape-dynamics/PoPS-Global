import multiprocessing
import subprocess
import os
 
def execute_model_runs(run_iter):
    run_num = 2
    print('iteration: ', run_iter)
    subprocess.call(f'model.py {run_num} {run_iter}', shell=True)
    # print(os.system(f'python model.py {run_num} {run_iter}'))

run_num = 2
p = multiprocessing.Pool()
results = p.map(execute_model_runs, range(5))

p.close()

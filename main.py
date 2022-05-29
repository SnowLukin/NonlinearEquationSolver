from mpi4py import MPI
import time


def f(x):
    return x * x * x - 18 * x - 83


def rectangle_rule(func, a, b, _amount_of_iterations, multiplier):
    step = (b - a) / _amount_of_iterations
    result_sum = 0.0
    x_start = a + multiplier * step
    for i in range(1, int(_amount_of_iterations)):
        result_sum += func(x_start + i * step)
    return result_sum * step


def midpoint_rectangle_rule(func, start, end, _amount_of_iterations):
    return rectangle_rule(func, start, end, _amount_of_iterations, 0.5)


def secant_method(start_segment, end_segment, epsilon, _integral_from, _integral_result, _amount_of_iterations):
    while abs(end_segment-start_segment) >= epsilon:
        local_amount_of_iterations = _amount_of_iterations / p

        step_fx1 = (end_segment - _integral_from) / _amount_of_iterations
        local_start_fx1 = _integral_from + my_rank * local_amount_of_iterations * step_fx1
        local_end_fx1 = local_start_fx1 + local_amount_of_iterations * step_fx1

        step_fx2 = (start_segment - _integral_from) / _amount_of_iterations
        local_start_fx2 = _integral_from + my_rank * local_amount_of_iterations * step_fx2
        local_end_fx2 = local_start_fx2 + local_amount_of_iterations * step_fx2

        integral_fx1 = midpoint_rectangle_rule(f, local_start_fx1, local_end_fx1, local_amount_of_iterations)
        integral_fx2 = midpoint_rectangle_rule(f, local_start_fx2, local_end_fx2, local_amount_of_iterations)
        if my_rank == 0:
            fx1 = integral_fx1
            fx2 = integral_fx2
            for source in range(1, p):
                integral_array = comm.recv(source=source)
                fx1 = fx1 + integral_array[0]
                fx2 = fx2 + integral_array[1]
            fx1 -= _integral_result
            fx2 -= _integral_result
            x_temp = end_segment
            end_segment = end_segment - (end_segment - start_segment) * fx1 / (fx1 - fx2)
            start_segment = x_temp
            for rank_index in range(1, p):
                comm.send([start_segment, end_segment], dest=rank_index)
        else:
            comm.send([integral_fx1, integral_fx2], dest=0)
            segment_array = comm.recv(source=0)
            start_segment = segment_array[0]
            end_segment = segment_array[1]
    return end_segment


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()
start_time = time.perf_counter()
integral_from = 8
integral_result = 10
amount_of_iterations = 240000
epsilon = 0.00001
x0 = 0  # initial approximation
x1 = x0 + 0.5
result = secant_method(x0, x1, epsilon, integral_from, integral_result, amount_of_iterations)
end_time = time.perf_counter()
if my_rank == 0:
    print(f"Result: {result}")
    print(f"Time: {end_time - start_time}")
MPI.Finalize


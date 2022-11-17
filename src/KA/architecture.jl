
norm(x) = sqrt(mapreduce(x->x*x, +, x))
synchronize(device) = wait(Event(device))

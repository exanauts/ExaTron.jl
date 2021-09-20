
norm(x) = mapreduce(x->x*x, +, x)
synchronize(device::KA.Device) = wait(Event(device))

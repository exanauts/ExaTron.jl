"""
Subroutine insort

Given an integer array keys of length n, this subroutine uses
an insertion sort to sort the keys in increasing order.

MINPACK-2 Project. March 1998.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function insort(n,keys)
    for j=2:n
        ind = keys[j]
        i = j - 1
        while (i > 0) && (keys[i] > ind)
            keys[i+1] = keys[i]
            i = i - 1
        end
        keys[i+1] = ind
    end

    return
end

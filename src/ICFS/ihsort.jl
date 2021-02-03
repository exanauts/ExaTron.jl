"""
Subroutine ihsort

Given an integer array keys of length n, this subroutine uses
a heap sort to sort the keys in increasing order.

This subroutine is a minor modification of code written by
Mark Jones and Paul Plassmann.

MINPACK-2 Project. March 1998.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function ihsort(n,keys)
    if  n <= 1
        return
    end

    # Build the heap.

    mid = floor(Int,n/2)
    for k=mid:-1:1
        x = keys[k]
        lheap = k
        rheap = n
        m = lheap*2
        while m <= rheap
            if m < rheap
                if keys[m] < keys[m+1]
                    m = m + 1
                end
            end
            if x >= keys[m]
                m = rheap + 1
            else
                keys[lheap] = keys[m]
                lheap = m
                m = 2*lheap
            end
        end
        keys[lheap] = x
    end

    # Sort the heap.

    for k=n:-1:2
        x = keys[k]
        keys[k] = keys[1]
        lheap = 1
        rheap = k-1
        m = 2
        while m <= rheap
            if m < rheap
                if keys[m] < keys[m+1]
                    m = m+1
                end
            end
            if x >= keys[m]
                m = rheap + 1
            else
                keys[lheap] = keys[m]
                lheap = m
                m = 2*lheap
            end
        end
        keys[lheap] = x
    end

    return
end

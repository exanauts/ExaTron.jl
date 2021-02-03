"""
Subroutine dsel2

Given an array x, this subroutine permutes the elements of the
array keys so that

  abs(x(keys(i))) <= abs(x(keys(k))),  1 <= i <= k,
  abs(x(keys(k))) <= abs(x(keys(i))),  k <= i <= n.

In other words, the smallest k elements of x in absolute value are
x(keys(i)), i = 1,...,k, and x(keys(k)) is the kth smallest element.

MINPACK-2 Project. March 1998.
Argonne National Laboratory.
William D. Kastak, Chih-Jen Lin, and Jorge J. More'.

Revised October 1999. Length of x was incorrectly set to n.
"""
function dsel2(n,x,keys,k)
    if (n <= 1 || k <= 0 || k > n)
        return
    end

    u = n
    l = 1
    lc = n
    lp = 2*n

    while (l < u)

        # Choose the partition as the median of the elements in
        # positions l+s*(u-l) for s = 0, 0.25, 0.5, 0.75, 1.
        # Move the partition element into position l.

        p1 = floor(Int,(u+3*l)/4)
        p2 = floor(Int,(u+l)/2)
        p3 = floor(Int,(3*u+l)/4)

        # Order the elements in positions l and p1.

        if abs(x[keys[l]]) > abs(x[keys[p1]])
            swap = keys[l]
            keys[l] = keys[p1]
            keys[p1] = swap
        end

        # Order the elements in positions p2 and p3.

        if abs(x[keys[p2]]) > abs(x[keys[p3]])
            swap = keys[p2]
            keys[p2] = keys[p3]
            keys[p3] = swap
        end

        # Swap the larger of the elements in positions p1
        # and p3, with the element in position u, and reorder
        # the first two pairs of elements as necessary.

        if abs(x[keys[p3]]) > abs(x[keys[p1]])
            swap = keys[p3]
            keys[p3] = keys[u]
            keys[u] = swap
            if abs(x[keys[p2]]) > abs(x[keys[p3]])
                swap = keys[p2]
                keys[p2] = keys[p3]
                keys[p3] = swap
            end
        else
            swap = keys[p1]
            keys[p1] = keys[u]
            keys[u] = swap
            if abs(x[keys[l]]) > abs(x[keys[p1]])
                swap = keys[l]
                keys[l] = keys[p1]
                keys[p1] = swap
            end
        end

        # If we define a(i) = abs(x(keys(i))) for i = 1,...,n, we have
        # permuted keys so that
        #
        #  a(l) <= a(p1), a(p2) <= a(p3), max(a(p1),a(p3)) <= a(u).
        #
        # Find the third largest element of the four remaining
        # elements (the median), and place in position l.

        if abs(x[keys[p1]]) > abs(x[keys[p3]])
            if abs(x[keys[l]]) <= abs(x[keys[p3]])
                swap = keys[l]
                keys[l] = keys[p3]
                keys[p3] = swap
            end
        else
            if abs(x[keys[p2]]) <= abs(x[keys[p1]])
                swap = keys[l]
                keys[l] = keys[p1]
                keys[p1] = swap
            else
                swap = keys[l]
                keys[l] = keys[p2]
                keys[p2] = swap
            end
        end

        # Partition the array about the element in position l.

        m = l
        for i=l+1:u
            if abs(x[keys[i]]) < abs(x[keys[l]])
                m = m + 1
                swap = keys[m]
                keys[m] = keys[i]
                keys[i] = swap
            end
        end

        # Move the partition element into position m.

        swap = keys[l]
        keys[l] = keys[m]
        keys[m] = swap

        # Adjust the values of l and u.

        if k >= m
            l = m + 1
        end
        if k <= m
            u = m - 1
        end

        # Check fo multiple medians if the length of the subarray
        # has not decreased by 1/3 after two consecutive iterations.

        if (3*(u-l) > 2*lp) && (k > m)

            # Partition the remaining elements into those elements
            # equal to x(m), and those greater than x(m). Adjust
            # the values of l and u.

            p = m
            for i=m+1:u
                if abs(x[keys[i]]) == abs(x[keys[m]])
                    p = p + 1
                    swap = keys[p]
                    keys[p] = keys[i]
                    keys[i] = swap
                end
            end
            l = p + 1
            if k <= p
                u = p - 1
            end
        end

        # Update the length indicators for the subarray.

        lp = lc
        lc = u-l
    end

    return
end
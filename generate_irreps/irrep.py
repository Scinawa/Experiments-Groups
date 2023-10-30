from sage.all import *
import cnine
from snob import *
import numpy as np
import multiprocessing
from datetime import datetime
import sys

representations_folder = '/hpctmp/quantumwuongo/data/representations/'

N = int(sys.argv[2])

group = SymmetricGroup(N)

def transversal_sn(group_n: int, subgroup_n: int) -> list:
    assert group_n > subgroup_n
    if group_n - subgroup_n == 2:
        representatives = list()
        for i in range(group_n):
            for j in range(0, group_n):
                if i != j:
                    representative = [0] * group_n
                    representative[group_n - 2] = i + 1
                    representative[group_n - 1] = j + 1
                    q = {group_n - 1, group_n} - {i + 1, j + 1}
                    for k in range(0, group_n - 2):
                        if k != i and k != j:
                            representative[k] = k + 1
                        elif k == i:
                            if len(q) == 1:
                                representative[k] = q.pop()
                            else:
                                representative[k] = group_n - 1
                                q.remove(group_n - 1)
                        elif k == j:
                            if len(q) == 1:
                                representative[k] = q.pop()
                            else:
                                representative[k] = group_n
                                q.remove(group_n)
                    representatives.append(representative)
        return representatives
    else:
        raise NotImplementedError("Generating transversal is implemented only for S_n / S_{n-2}.")


def rep_sn(group_element, rep_part: list[int] | tuple[int]):
    """Returns the representation matrix of group element."""
    rep_part = list(rep_part)
    # trivial representation
    if len(rep_part) == 1:
        return
    # nontrivial representation
    n_folder = representations_folder + str(sum(rep_part)) + '/'
    group_element_name = str(group(group_element))
    representation_path = n_folder + str(rep_part) + group_element_name + '.npy'
    if not os.path.exists(representation_path):
        if not os.path.exists(n_folder):
            os.mkdir(n_folder)
        rho=SnIrrep(rep_part)
        result = np.array(rho[group_element].torch())
        np.save(representation_path, result)
    return


def generate_parallel(parts, transversal):
    number_of_threads = multiprocessing.cpu_count()
    p = multiprocessing.get_context("fork").Pool(number_of_threads)
    p.starmap(rep_sn, [(g, part.to_list()) for part in parts for g in transversal])


if __name__ == '__main__':
    if len(sys.argv) < 2 or (sys.argv[1] != '1' and sys.argv[1] != '2'):
        print("The program takes an argument: 1 for iterative; 2 for parallel.")
        exit()

    print(f"N={N}:\t", end='')
    beginning = datetime.now()

    transversal = transversal_sn(N, N - 2)
    #transversal = SymmetricGroup(N)
    if N >= 2:
        parts = Partitions(N, inner=[N - 2]).list()
    else:
        parts = Partitions(N).list()

    parts.remove([N])

    if sys.argv[1] == '1':
        print("iterative")
        for part in parts:
            for g in transversal:
                print("{}({})".format(part, g))
                rep_sn(g, part)
    elif sys.argv[1] == '2':
        print(type(parts), type(transversal))
        print("parallel")
        generate_parallel(parts, transversal)

    end = datetime.now()

    duration = end - beginning
    duration_in_s = duration.total_seconds()

    print("It took {}".format(duration_in_s))

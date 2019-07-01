import sys

sys.setrecursionlimit(2000)

def lcs_matrix(s1, s2):
    matrix=[[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i == 0 or j == 0:
                    matrix[i][j] = 0
            elif s1[i-1] == s2[j-1]:
                matrix[i][j] = matrix[i-1][j-1] + 1
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1])
    return matrix

def indexes_backtrack_lcs_matrix(matrix, s1, s2, i, j):
    if i == 0 or j == 0:
        return []

    if s1[i-1] == s2[j-1]:
        return indexes_backtrack_lcs_matrix(matrix, s1, s2, i-1, j-1) + [(i-1, j-1)]
    if matrix[i][j-1] > matrix[i-1][j]:
        return indexes_backtrack_lcs_matrix(matrix, s1, s2, i, j-1)
    return indexes_backtrack_lcs_matrix(matrix, s1, s2, i-1, j)

def backtrack_lcs_matrix(matrix, s1, s2, i, j):
    if i == 0 or j == 0:
        return []

    if s1[i-1] == s2[j-1]:
        return backtrack_lcs_matrix(matrix, s1, s2, i-1, j-1) + [s1[i-1]]
    if matrix[i][j-1] > matrix[i-1][j]:
        return backtrack_lcs_matrix(matrix, s1, s2, i, j-1)
    return backtrack_lcs_matrix(matrix, s1, s2, i-1, j)

def all_backtrack_lcs_matrix(matrix, s1, s2, i, j):
    if i == 0 or j == 0:
        return [[]]
    elif s1[i-1] == s2[j-1]:
        return [s + [s1[i-1]] for s in all_backtrack_lcs_matrix(matrix,
                                                                s1, s2,
                                                                i-1, j-1)]
    else:
        result = []
        if matrix[i][j-1] >= matrix[i-1][j]:
            result = result + all_backtrack_lcs_matrix(matrix,
                                                       s1, s2,
                                                       i, j-1)
        if matrix[i-1][j] >= matrix[i][j-1]:
            result = result + all_backtrack_lcs_matrix(matrix,
                                                       s1, s2,
                                                       i-1, j)
        return result

def lcs(s1, s2, joiner=""):
    matrix = lcs_matrix(s1, s2)
    if joiner ==  "":
        return backtrack_lcs_matrix(matrix, s1, s2, len(s1), len(s2)) 
    return joiner.join(backtrack_lcs_matrix(matrix, s1, s2, len(s1), len(s2)))

def lcs_indexes(s1, s2, joiner=""):
    matrix = lcs_matrix(s1, s2)
    if joiner ==  "":
        return indexes_backtrack_lcs_matrix(matrix, s1, s2, len(s1), len(s2)) 
    return joiner.join(indexes_backtrack_lcs_matrix(matrix, s1, s2, len(s1), len(s2)))

def all_lcs(s1, s2, joiner=""):
    matrix = lcs_matrix(s1, s2)
    result = []
    for elem in all_backtrack_lcs_matrix(matrix, s1, s2, len(s1), len(s2)):
        result.append(joiner.join(elem))
    return list(set(result))

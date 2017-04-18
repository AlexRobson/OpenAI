def GroupDataByLength(X, y):

    N = []
    for i in X:
        N.append(len(X))

    counts = Counter(N)
    # TODO: Set up a grouping in the number of items for batching
#    for count in counts.keys():
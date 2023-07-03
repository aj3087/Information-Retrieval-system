def returnPrecisionK(relevanceJudgements):
    '''
    From an array of relevenace judgements, 1-> relevant , 0-> non relevant
    It computes precision.
    '''
    return sum(relevanceJudgements)/len(relevanceJudgements)

#
# Author: TugRulz
#
#
# inputs: size:total number of full and partial sketches, isFull: full or partial ? matrix , classId: class of sketch? matrix
# output: constraint matrix (must or cannot link)
import numpy
def getConstraints(size, isFull, classId):
    raise DeprecationWarning
    """Output : the constraints matrix of size len(class) x len(class) """
    FULL = 1
    MUST_LINK = 1
    CANNOT_LINK = -1
    DIAGONAL = 0
    cons = numpy.zeros((size, size))
    for index, el in enumerate(cons):
        if (isFull[index] == FULL):
            for index2, el2 in enumerate(el):
                if (index == index2 ):
                    cons[index][index2] = DIAGONAL
                else:
                    if (isFull[index2] == FULL):
                        if(classId[index] == classId[index2]):
                            cons[index][index2] = MUST_LINK
                        else:
                            cons[index][index2] = CANNOT_LINK
    return cons


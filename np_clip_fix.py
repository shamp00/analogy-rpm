#%%
import numpy as np
from numba import types
from numba.extending import overload

# pylint: disable-msg=E1111 
# E1111: Assigning to function call which doesn't return :) 

def _alloc_out(a, X):
    # dummy function that takes the arg we need to resolved (X)
    # and the information needed to do so (a)
    pass
    
@overload(_alloc_out)
def _alloc_out_impl(a, X):
    # overload the dummy function to return empty_like(a) if X is None else X
    # with type-specific implementations, in order to fully determine type of X
    # https://github.com/numba/numba/pull/3468#issuecomment-437974147
    if X is None or isinstance(X, types.NoneType):
        def impl(a, X):
            return np.empty_like(a)
    else:
        def impl(a, X):
            return X
    return impl

@overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    a_min_is_none = a_min is None or isinstance(a_min, types.NoneType)
    a_max_is_none = a_max is None or isinstance(a_max, types.NoneType)

    if a_min_is_none and a_max_is_none:
        def np_clip_impl(a, a_min, a_max, out=None):
            # implementation for no bounds
            raise ValueError("array_clip: must set either max or min")

    elif a_min_is_none:
        def np_clip_impl(a, a_min, a_max, out=None):
            # implementation for upper-bound only
            ret = _alloc_out(a, out)
            for index, val in np.ndenumerate(a):
                if val > a_max:
                    ret[index] = a_max
                else:
                    ret[index] = val
            return ret

    elif a_max_is_none:
        def np_clip_impl(a, a_min, a_max, out=None):
            # implementation for lower-bound only
            ret = _alloc_out(a, out)
            for index, val in np.ndenumerate(a):
                if val < a_min:
                    ret[index] = a_min
                else:
                    ret[index] = val
            return ret

    else:
        def np_clip_impl(a, a_min, a_max, out=None):
            # implementation when both bounds defined
            ret = _alloc_out(a, out)
            for index, val in np.ndenumerate(a):
                if val < a_min:
                    ret[index] = a_min
                elif val > a_max:
                    ret[index] = a_max
                else:
                    ret[index] = val
            return ret

    return np_clip_impl

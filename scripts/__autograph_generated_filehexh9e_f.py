# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf___check_and_reshape_inputs(self, inputs):
            with ag__.FunctionScope('_check_and_reshape_inputs', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                shape = ag__.converted_call(ag__.ld(inputs).get_shape, (), None, fscope)
                ndim = ag__.converted_call(ag__.ld(len), (ag__.ld(shape),), None, fscope)
                ag__.converted_call(ag__.ld(tf).debugging.assert_greater_equal, (ag__.ld(ndim), 2, f'Expected min_ndim=2, found ndim={ag__.ld(ndim)}. Full shape received: {ag__.ld(shape)}'), None, fscope)

                def get_state():
                    return ()

                def set_state(block_vars):
                    pass

                def if_body():
                    ag__.converted_call(ag__.ld(tf).debugging.assert_equal, (ag__.ld(inputs).shape[-1], ag__.ld(self).in_size, f'Expected last dimension of inputs to be {ag__.ld(self).in_size}, found {ag__.ld(shape)[-1]}.'), None, fscope)

                def else_body():
                    pass
                ag__.if_stmt(ag__.ld(inputs).shape[-1] != None, if_body, else_body, get_state, set_state, (), 0)
                orig_shape = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(inputs),), None, fscope)[:-1]

                def get_state_1():
                    return (inputs,)

                def set_state_1(vars_):
                    nonlocal inputs
                    inputs, = vars_

                def if_body_1():
                    nonlocal inputs
                    inputs = ag__.converted_call(ag__.ld(tf).reshape, (ag__.ld(inputs), (-1, ag__.ld(self).in_size)), None, fscope)

                def else_body_1():
                    nonlocal inputs
                    pass
                ag__.if_stmt(ag__.ld(inputs).shape[-1] != None, if_body_1, else_body_1, get_state_1, set_state_1, ('inputs',), 1)
                try:
                    do_return = True
                    retval_ = (ag__.ld(inputs), ag__.ld(orig_shape))
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___check_and_reshape_inputs
    return inner_factory
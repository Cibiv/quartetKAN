# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, inputs, *args, **kwargs):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                inputs, orig_shape = ag__.converted_call(ag__.ld(self)._check_and_reshape_inputs, (ag__.ld(inputs),), None, fscope)
                output_shape = ag__.converted_call(ag__.ld(tf).concat, ([ag__.ld(orig_shape), [ag__.ld(self).units]],), dict(axis=0), fscope)
                spline_out = ag__.converted_call(ag__.ld(self).calc_spline_output, (ag__.ld(inputs),), None, fscope)
                spline_out = ag__.ld(spline_out)
                spline_out += ag__.converted_call(tf.expand_dims, (ag__.converted_call(self.basis_activation, (inputs,), None, fscope),), dict(axis=-1), fscope)
                spline_out = ag__.ld(spline_out)
                spline_out *= ag__.converted_call(tf.expand_dims, (self.scale_factor,), dict(axis=0), fscope)
                spline_out = ag__.converted_call(ag__.ld(tf).reshape, (ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.ld(spline_out),), dict(axis=-2), fscope), ag__.ld(output_shape)), None, fscope)

                def get_state():
                    return (spline_out,)

                def set_state(vars_):
                    nonlocal spline_out
                    spline_out, = vars_

                def if_body():
                    nonlocal spline_out
                    spline_out = ag__.ld(spline_out)
                    spline_out += self.bias

                def else_body():
                    nonlocal spline_out
                    pass
                ag__.if_stmt(ag__.ld(self).use_bias, if_body, else_body, get_state, set_state, ('spline_out',), 1)
                try:
                    do_return = True
                    retval_ = ag__.ld(spline_out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory
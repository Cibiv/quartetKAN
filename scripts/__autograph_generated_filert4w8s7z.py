# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__calc_spline_output(self, inputs):
            """
        calculate the spline output, each feature of each sample is mapped to `out_size` features,         using `out_size` different B-spline basis functions, so the output shape is `(batch_size, in_size, out_size)`

        Parameters
        ----------
        inputs : tf.Tensor
            the input tensor with shape `(batch_size, in_size)`
        
        Returns
        -------
        spline_out : tf.Tensor
            the output tensor with shape `(batch_size, in_size, out_size)`
        """
            with ag__.FunctionScope('calc_spline_output', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                spline_in = ag__.converted_call(ag__.ld(calc_spline_values), (ag__.ld(inputs), ag__.ld(self).grid, ag__.ld(self).spline_order), None, fscope)
                spline_out = ag__.converted_call(ag__.ld(tf).einsum, ('bik,iko->bio', ag__.ld(spline_in), ag__.ld(self).spline_kernel), None, fscope)
                try:
                    do_return = True
                    retval_ = ag__.ld(spline_out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__calc_spline_output
    return inner_factory
# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf___update_step_xla(self, gradient, variable, key):
            """A wrapper of `update_step` to enable XLA acceleration.

        Due to `tf.function` tracing mechanism, for (gradient, variable) pairs
        of the same shape and dtype, the execution graph always invoke the first
        pair it has seen. Thus, we need a `key` argument to make each (gradient,
        variable) pair unique. In additions, XLA cannot understand string input,
        so the key is an integer.

        Args:
          gradient: backpropagated gradient of the given variable.
          variable: variable whose value needs to be updated.
          key (int): a unique key that identifies the variable.

        Returns:
          An `Operation` that applies the specified gradients.
        """
            with ag__.FunctionScope('_update_step_xla', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(self)._update_step, (ag__.ld(gradient), ag__.ld(variable)), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___update_step_xla
    return inner_factory
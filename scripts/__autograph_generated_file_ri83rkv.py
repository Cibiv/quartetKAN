# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__calc_spline_values(x: tf.Tensor, grid: tf.Tensor, spline_order: int):
            """
    Calculate B-spline values for the input tensor.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor with shape (batch_size, in_size).
    grid : tf.Tensor
        The grid tensor with shape (in_size, grid_size + 2 * spline_order + 1).
    spline_order : int
        The spline order.

    Returns: tf.Tensor
        B-spline bases tensor of shape (batch_size, in_size, grid_size + spline_order).
    """
            with ag__.FunctionScope('calc_spline_values', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                assert ag__.converted_call(ag__.ld(len), (ag__.converted_call(ag__.ld(x).get_shape, (), None, fscope),), None, fscope) == 2
                x = ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(x),), dict(axis=-1), fscope)
                bases = ag__.converted_call(ag__.ld(tf).logical_and, (ag__.converted_call(ag__.ld(tf).greater_equal, (ag__.ld(x), ag__.ld(grid)[:, :-1]), None, fscope), ag__.converted_call(ag__.ld(tf).less, (ag__.ld(x), ag__.ld(grid)[:, 1:]), None, fscope)), None, fscope)
                bases = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(bases), ag__.ld(x).dtype), None, fscope)

                def get_state():
                    return (bases,)

                def set_state(vars_):
                    nonlocal bases
                    bases, = vars_

                def loop_body(itr):
                    nonlocal bases
                    k = itr
                    bases = (ag__.ld(x) - ag__.ld(grid)[:, :-(ag__.ld(k) + 1)]) / (ag__.ld(grid)[:, ag__.ld(k):-1] - ag__.ld(grid)[:, :-(ag__.ld(k) + 1)]) * ag__.ld(bases)[:, :, :-1] + (ag__.ld(grid)[:, ag__.ld(k) + 1:] - ag__.ld(x)) / (ag__.ld(grid)[:, ag__.ld(k) + 1:] - ag__.ld(grid)[:, 1:-ag__.ld(k)]) * ag__.ld(bases)[:, :, 1:]
                k = ag__.Undefined('k')
                ag__.for_stmt(ag__.converted_call(ag__.ld(range), (1, ag__.ld(spline_order) + 1), None, fscope), None, loop_body, get_state, set_state, ('bases',), {'iterate_names': 'k'})
                try:
                    do_return = True
                    retval_ = ag__.ld(bases)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__calc_spline_values
    return inner_factory
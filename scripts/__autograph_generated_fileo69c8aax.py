# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__get_record_defaults(self):
            with ag__.FunctionScope('get_record_defaults', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                zeros = ag__.converted_call(ag__.ld(tf).zeros, (), dict(shape=(1,), dtype=ag__.ld(tf).float32), fscope)
                ones = ag__.converted_call(ag__.ld(tf).ones, (), dict(shape=(1,), dtype=ag__.ld(tf).float32), fscope)
                try:
                    do_return = True
                    retval_ = [ag__.ld(zeros)] * (ag__.ld(self).layers[0] + ag__.ld(self).offset) + [ag__.ld(ones)]
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__get_record_defaults
    return inner_factory
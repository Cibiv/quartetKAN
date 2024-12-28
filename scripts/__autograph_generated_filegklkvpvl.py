# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__parse_row(self, tf_string):
            with ag__.FunctionScope('parse_row', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                data = ag__.converted_call(ag__.ld(tf).io.decode_csv, (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(tf_string),), dict(axis=0), fscope), ag__.converted_call(ag__.ld(self).get_record_defaults, (), None, fscope)), None, fscope)
                features = ag__.ld(data)[ag__.ld(self).offset:-1]
                features = ag__.converted_call(ag__.ld(tf).stack, (ag__.ld(features),), dict(axis=-1), fscope)
                label = ag__.ld(data)[-1]
                features = ag__.converted_call(ag__.ld(tf).squeeze, (ag__.ld(features),), dict(axis=0), fscope)
                label = ag__.converted_call(ag__.ld(tf).squeeze, (ag__.ld(label),), dict(axis=0), fscope)
                try:
                    do_return = True
                    retval_ = (ag__.ld(features), ag__.ld(label))
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__parse_row
    return inner_factory